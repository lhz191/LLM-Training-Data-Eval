#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成数据质量 (Generation Quality) — 合成/生成数据专有指标

评估由 LLM 或 pipeline 合成的 agent 训练数据的质量。

=== 背景 ===
越来越多的 agent 训练数据是合成生成的（APIGen-MT, Toucan, AgentInstruct 等）。
合成数据的优势是量大、可控，但质量参差不齐。
本指标专门评估合成数据中常见的质量问题，帮助筛选高质量训练样本。

适用于单轮和多轮数据。

=== 检测维度 ===

1. 轨迹自然度 (Trajectory Naturalness)
   agent 的操作序列是否像真实人类操作，还是有明显的"机器味"。
   - 不自然的模式：每步都先滚动再点击（固定模板）、
     action_repr 格式过于统一（真实数据有变化）。
   - 检测方法：与真实数据集的 action 分布做 KL 散度比较。

2. 指令-轨迹对齐度 (Instruction-Trajectory Alignment)
   instruction 说的任务和 action 序列实际做的是否一致。
   - 例：instruction 说 "搜索机票"，但 action 全是在看酒店。
   - 检测方法：用 LLM 判断 instruction 和 action_reprs 的语义一致性，
     或用 NLI 模型判断蕴含关系。

3. 幻觉检测 (Hallucination Detection)
   agent 是否在操作中引用了不存在的元素或捏造了结果。
   - 例：target_element 引用了一个页面上不存在的按钮。
   - 例：action_repr 中描述了实际未发生的操作。
   - 检测方法：如果有 cleaned_html，检查 target_element 是否存在于 HTML 中。

4. 多样性衰减 (Diversity Decay)
   合成数据是否存在模式坍缩——大量样本的轨迹高度相似。
   - 检测方法：计算样本间 action 序列的编辑距离分布，
     与真实数据集对比。

5. 轮次冗余度 (Turn Redundancy) [多轮专有]
   多轮会话中是否存在不必要的轮次（对任务完成无贡献）。
   - 例：agent 反复确认同一件事、重复执行相同操作。
   - 检测方法：计算相邻轮次的 action 序列相似度。

=== 使用方式 ===
    from mt_text_gui_agent_eval.data_types import Session, Record
    from mt_text_gui_agent_eval.metrics.generation_quality import (
        compute_generation_quality, compute_generation_quality_multiturn
    )

    # 单轮数据
    results = compute_generation_quality(
        data_iterator=loader.iterate(),
        dataset_name='synthetic_web_agent',
    )

    # 多轮数据
    results = compute_generation_quality_multiturn(
        session_iterator=iter(sessions),
        dataset_name='APIGen-MT',
    )
"""

import os
import sys
from typing import Iterator, Dict, Any, Optional

_this_dir = os.path.dirname(os.path.abspath(__file__))
_mt_dir = os.path.dirname(_this_dir)

if _mt_dir not in sys.path:
    sys.path.insert(0, _mt_dir)

from data_types import Session, Record  # noqa: E402


def compute_generation_quality(
    data_iterator: Iterator[Record],
    dataset_name: str,
    reference_data: Optional[Iterator[Record]] = None,
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    计算（单轮）合成数据的生成质量

    Args:
        data_iterator: 待评估数据
        dataset_name: 数据集名称
        reference_data: 真实数据作为参照（用于分布对比），可选
        output_file: 输出文件路径
        max_samples: 最大样本数

    Returns:
        {
            'dataset': str,
            'total_samples': int,
            'trajectory_naturalness_score': float,      # 轨迹自然度 [0, 1]
            'instruction_alignment_score': float,       # 指令-轨迹对齐度 [0, 1]
            'hallucination_rate': float,                # 幻觉率 [0, 1]（越低越好）
            'diversity_decay_score': float,             # 多样性衰减 [0, 1]（越高越好）
            'overall_quality_score': float,
            'flagged_samples': [...]                    # 被标记为低质量的样本
        }
    """
    # TODO: 实现生成数据质量评估
    # 思路：
    # 1. trajectory_naturalness:
    #    - 提取 action_type 序列的 n-gram 分布
    #    - 如果有 reference_data，计算与真实数据 action 分布的 KL 散度
    #    - 检测固定模板模式（连续 N 个样本的 action_type 序列完全相同）
    #
    # 2. instruction_alignment:
    #    - 提取 instruction 的关键词
    #    - 检查 action_repr 序列中是否包含相关操作
    #    - 高级：用 NLI 模型或 LLM 判断蕴含关系
    #
    # 3. hallucination_detection:
    #    - 如果有 cleaned_html 和 target_element，检查元素是否存在于 HTML
    #    - 检查 action_value 是否合理（如日期格式、价格范围）
    #
    # 4. diversity_decay:
    #    - 计算样本间 action_type 序列的编辑距离
    #    - 如果大量样本距离很小，说明存在模式坍缩
    raise NotImplementedError(
        "generation_quality 尚未实现。"
        "需要合成数据集实际接入后，根据数据特点设计具体检测逻辑。"
    )


def compute_generation_quality_multiturn(
    session_iterator: Iterator[Session],
    dataset_name: str,
    reference_sessions: Optional[Iterator[Session]] = None,
    output_file: Optional[str] = None,
    max_sessions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    计算多轮合成数据的生成质量（在单轮基础上增加轮次冗余度检测）

    Args:
        session_iterator: 待评估的 Session 迭代器
        dataset_name: 数据集名称
        reference_sessions: 真实多轮数据作为参照，可选
        output_file: 输出文件路径
        max_sessions: 最大会话数

    Returns:
        单轮指标 + {
            'turn_redundancy_rate': float,  # 轮次冗余率 [0, 1]（越低越好）
        }
    """
    # TODO: 实现多轮生成数据质量评估
    # 在单轮指标基础上，增加：
    # 1. turn_redundancy: 计算相邻轮次的 action_repr 集合的 Jaccard 相似度，
    #    高相似度说明两轮做了相似的事情，可能是冗余轮次
    # 2. 提取所有 Record 调用 compute_generation_quality 获取单轮基础指标
    raise NotImplementedError(
        "generation_quality_multiturn 尚未实现。"
        "需要多轮合成数据集实际接入后，根据数据特点设计具体检测逻辑。"
    )
