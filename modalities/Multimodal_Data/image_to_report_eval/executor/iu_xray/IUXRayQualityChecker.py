#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IU X-Ray 报告质量评估器

继承链：
  common.base.BaseQualityChecker
    └── report_executor.QualityChecker  (modality='image_to_report')
          └── IUXRayQualityChecker

使用 LLM Judge（文本模型）评估放射学报告的质量。

数据集特征（评估维度据此设计）：
- 报告为段落式自由文本，几乎无 Findings/Impression 等分段标题
- 中位长度 ~200 字符、~5 句话，是典型的简洁放射学描述风格
- 约 44% 报告含 XXXX 占位符（去标识化/OCR 产物，替换了原文中的某些词）
- 所有样本共享同一 instruction（固定模板），质量评估聚焦于报告本身

评估维度：
1. 医学术语规范性 (Terminology): 非 XXXX 部分是否使用标准放射学术语
2. 临床逻辑性 (Clinical Logic): 描述是否按解剖结构系统推进、发现与判断是否逻辑连贯
3. 信息密度 (Information Density): 在给定篇幅内传达的临床信息量是否充分
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
# LLM 配置（与 validity.py 共享同一平台）
# =============================================================================

LLM_API_KEY = 'sk-o0QqcwC8XNHU6gGT7CYdMSQGJQQMtjKJSqw6K9G21IaoOElt'
LLM_BASE_URL = 'http://35.220.164.252:3888/v1/'
LLM_MODEL = 'gpt-4o'


# =============================================================================
# Prompt
# =============================================================================

IU_XRAY_QUALITY_PROMPT = """你是一位资深放射科医师和医学数据质量评估专家。
请评估以下胸部 X 光报告的质量。

【重要背景】
该报告来自 IU X-Ray 数据集（Indiana University 胸部 X 光报告集），具有以下特征：
- 报告为段落式自由文本，通常不含 Findings / Impression 等正式分段标题
- "XXXX" 是去标识化占位符（替换了原文中的敏感词或 OCR 无法识别的词），请忽略 XXXX 本身，不要因为它扣分
- 对于正常检查，简短报告（如 "The lungs are clear. Heart size is normal."）是完全合理的

【报告内容】
{report}

请从以下三个维度评估报告质量（每个维度 1-5 分）：

1. **医学术语规范性 (Terminology)**
   评估非 XXXX 部分的术语使用
   - 5: 全部使用标准放射学术语（如 cardiomegaly, opacity, effusion, consolidation, atelectasis）
   - 3: 大部分使用专业术语，偶有口语化表达
   - 1: 缺少专业术语，表达过于口语化或使用错误术语

2. **临床逻辑性 (Clinical Logic)**
   评估描述是否具有系统性和逻辑连贯性（不要求有正式分段标题）
   - 5: 按解剖结构系统性描述（如心脏→肺野→纵隔→骨骼），发现与临床判断逻辑一致
   - 3: 有一定顺序但不够系统，或部分发现缺少对应的临床解释
   - 1: 描述散乱无序，发现之间缺乏逻辑关联

3. **信息密度 (Information Density)**
   评估在给定篇幅内传达的有效临床信息量（正常检查的简短报告不应被惩罚）
   - 5: 每句话都传达明确的临床信息，无冗余重复；正常和异常发现都有恰当覆盖
   - 3: 有一定信息量但存在明显冗余或关键信息遗漏
   - 1: 内容空洞、大量重复，或重要临床信息严重缺失

请以 JSON 格式输出：
```json
{{
    "terminology": {{"score": 1-5, "reason": "简要说明"}},
    "clinical_logic": {{"score": 1-5, "reason": "简要说明"}},
    "information_density": {{"score": 1-5, "reason": "简要说明"}}
}}
```

只输出 JSON，不要其他内容。"""


class IUXRayQualityChecker(QualityChecker):
    """
    IU X-Ray 报告质量评估器

    评估维度（基于数据集实际特征设计）：
    1. terminology (医学术语规范性): 1-5，忽略 XXXX 占位符
    2. clinical_logic (临床逻辑性): 1-5，不要求正式分段标题
    3. information_density (信息密度): 1-5，正常检查的简短报告不被惩罚

    通过门槛: 三个维度平均分 >= 3.0
    """

    PASS_THRESHOLD = 3.0
    DIMENSIONS = ["terminology", "clinical_logic", "information_density"]

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
        prompt = IU_XRAY_QUALITY_PROMPT.format(report=sample.report)

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
