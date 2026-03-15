#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShareGPT4V 报告质量评估器

继承链：
  common.base.BaseQualityChecker
    └── report_executor.QualityChecker  (modality='image_to_report')
          └── ShareGPT4VQualityChecker

ShareGPT4V report subset 的用途：
    作为多模态模型 instruction tuning 数据，教模型生成详尽的图像描述。
    由 GPT-4V (cap100k) 和 ShareCaptioner (captioner1246k) 生成。

描述风格特点（来自实际数据观察）：
    - 多段落、文学性叙述（"In the center of the image, a vibrant blue lunch tray..."）
    - 多角度覆盖：主体、背景、空间关系、颜色/纹理、动作/状态
    - 平均 800-900 字符

评估维度（面向 post-training 数据质量）：
1. 描述丰富度: 是否从多角度（主体、背景、空间、颜色、动作等）覆盖了视觉信息
2. 训练价值: 信息密度是否适合作为 instruction tuning 数据，能否帮助模型学会详细图像理解
3. 内容连贯性: 描述内部逻辑是否一致，有无自相矛盾或重复冗余
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import ImageToReportSample
from report_executor import QualityChecker


# =============================================================================
# LLM 配置
# =============================================================================

LLM_API_KEY = 'sk-o0QqcwC8XNHU6gGT7CYdMSQGJQQMtjKJSqw6K9G21IaoOElt'
LLM_BASE_URL = 'http://35.220.164.252:3888/v1/'
LLM_MODEL = 'gpt-4o'


# =============================================================================
# Prompt
# =============================================================================

SHAREGPT4V_QUALITY_PROMPT = """你是一个多模态训练数据质量评估专家。
以下是一段图像描述文本，来自用于训练多模态大模型的 instruction tuning 数据集。
该数据集的目标是教模型学会生成详尽、全面的图像描述。

【用户指令】
{instruction}

【图像描述】
{report}

请从以下三个维度评估该描述作为训练数据的质量（每个维度 1-5 分）：

1. **描述丰富度 (Richness)**
   该描述是否从多角度覆盖了视觉信息？
   - 5: 覆盖了主体、背景、空间关系、颜色/纹理、动作/状态等多个维度
   - 3: 覆盖了主体和部分背景，但缺少空间、颜色等细节
   - 1: 只有一两句概括性描述，几乎没有视觉细节

2. **训练价值 (Training Value)**
   该描述的信息密度是否适合作为 instruction tuning 数据？
   - 5: 信息密度高，描述具体且无废话，模型能从中学到丰富的图像理解能力
   - 3: 有一定信息量，但部分内容过于泛泛或有轻微冗余
   - 1: 信息量极低，或充斥大量无意义的套话和重复

3. **内容连贯性 (Coherence)**
   描述内部逻辑是否一致，组织是否合理？
   - 5: 结构清晰（如从整体到局部，从前景到背景），无矛盾无重复
   - 3: 基本连贯，但组织不够系统，或有少量重复
   - 1: 逻辑混乱，自相矛盾，或大段重复

请以 JSON 格式输出：
```json
{{
    "richness": {{"score": 1-5, "reason": "简要说明"}},
    "training_value": {{"score": 1-5, "reason": "简要说明"}},
    "coherence": {{"score": 1-5, "reason": "简要说明"}}
}}
```

只输出 JSON，不要其他内容。"""


class ShareGPT4VQualityChecker(QualityChecker):
    """
    ShareGPT4V 图像描述质量评估器

    评估维度（面向 post-training 数据质量）：
    1. richness (描述丰富度): 1-5
    2. training_value (训练价值): 1-5
    3. coherence (内容连贯性): 1-5

    通过门槛: 三个维度平均分 >= 3.0
    """

    PASS_THRESHOLD = 3.0
    DIMENSIONS = ["richness", "training_value", "coherence"]

    def __init__(
        self,
        api_key: str = LLM_API_KEY,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries

    def check(self, sample: ImageToReportSample) -> Dict[str, Any]:
        clean_instruction = sample.instruction.replace("<image>", "").strip()
        prompt = SHAREGPT4V_QUALITY_PROMPT.format(
            instruction=clean_instruction,
            report=sample.report,
        )

        response = self._call_llm(prompt)
        parsed = self._parse_response(response)

        scores = {}
        details = {}
        for dim in self.DIMENSIONS:
            dim_result = parsed.get(dim, {})
            scores[dim] = float(dim_result.get("score", 0.0))
            details[dim] = dim_result.get("reason", "")

        avg_score = sum(scores.values()) / len(scores) if scores else 0.0

        return {
            "scores": scores,
            "avg_score": avg_score,
            "passed": avg_score >= self.PASS_THRESHOLD,
            "details": details,
            "llm_error": parsed.get("_parse_error", False),
        }

    def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            from openai import OpenAI
        except ImportError:
            return None

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"  LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def _parse_response(self, response: Optional[str]) -> Dict[str, Any]:
        if not response:
            return {dim: {"score": 0.0, "reason": "LLM 调用失败"} for dim in self.DIMENSIONS} | {"_parse_error": True}

        try:
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text.strip())
        except Exception as e:
            return {dim: {"score": 0.0, "reason": f"解析失败: {e}"} for dim in self.DIMENSIONS} | {"_parse_error": True, "_raw": response}
