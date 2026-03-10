#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Turn Consistency -- 多轮专有指标

评估多轮会话中 agent 行为的前后一致性。

=== 检测维度 ===
1. 指令追踪一致性 (Instruction Tracking)
2. 操作不重复性 (Action Non-Redundancy)
3. 状态感知一致性 (State Awareness)
4. 目标连贯性 (Goal Coherence)

=== 使用方式 ===
    from mt_text_gui_agent_eval.data_types import Session
    from mt_text_gui_agent_eval.metrics.cross_turn_consistency import (
        compute_cross_turn_consistency,
    )

    results = compute_cross_turn_consistency(
        session_iterator=iter(sessions),
        dataset_name='WebLINX',
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


def compute_cross_turn_consistency(
    session_iterator: Iterator[Session],
    dataset_name: str,
    output_file: Optional[str] = None,
    max_sessions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    计算多轮会话的跨轮一致性

    Returns:
        {
            'dataset': str,
            'total_sessions': int,
            'instruction_tracking_score': float,
            'action_non_redundancy_score': float,
            'state_awareness_score': float,
            'goal_coherence_score': float,
            'overall_consistency_score': float,
            'per_session_results': [...]
        }
    """
    # TODO: 实现跨轮一致性检测
    # 思路：
    # 1. instruction_tracking: 比较相邻轮次的 instruction，检测修正/覆盖，
    #    检查后续 action 是否与最新 instruction 对齐（可用 LLM 做 NLI 判断）
    # 2. action_non_redundancy: 提取每轮的 action_repr 集合，检查跨轮重复
    # 3. state_awareness: 检查后续轮次是否在正确的状态基础上操作
    # 4. goal_coherence: 用 embedding 计算各轮 instruction 的语义相似度
    raise NotImplementedError(
        "cross_turn_consistency 尚未实现。"
        "需要多轮数据集（如 WebLINX）实际接入后，根据数据特点设计具体检测逻辑。"
    )
