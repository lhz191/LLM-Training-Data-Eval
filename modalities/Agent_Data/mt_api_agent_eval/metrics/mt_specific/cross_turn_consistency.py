#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Turn Consistency -- 多轮 API Agent 专有指标

评估多轮 API 会话中 agent 行为的前后一致性。

=== 检测维度 ===
1. 查询追踪一致性：用户修正查询后，agent 是否响应最新意图
2. 工具调用不重复性：agent 是否重复调用了已完成的操作
3. 参数一致性：跨轮引用的参数值（如 order_id）是否前后一致
4. 目标连贯性：多轮查询是否围绕连贯目标展开
"""

import os
import sys
from typing import Iterator, Dict, Any, Optional

_this_dir = os.path.dirname(os.path.abspath(__file__))
_mt_dir = os.path.dirname(_this_dir)

if _mt_dir not in sys.path:
    sys.path.insert(0, _mt_dir)

from data_types import Session  # noqa: E402


def compute_cross_turn_consistency(
    session_iterator: Iterator[Session],
    dataset_name: str,
    output_file: Optional[str] = None,
    max_sessions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    计算多轮 API 会话的跨轮一致性

    Returns:
        {
            'dataset': str,
            'total_sessions': int,
            'query_tracking_score': float,
            'call_non_redundancy_score': float,
            'param_consistency_score': float,
            'goal_coherence_score': float,
            'overall_consistency_score': float,
        }
    """
    # TODO: 实现
    # 1. query_tracking: 检测查询修正后 agent 的工具调用是否对齐
    # 2. call_non_redundancy: 跨轮工具调用去重检测
    # 3. param_consistency: 提取跨轮共享的参数（如 user_id），检查值是否一致
    # 4. goal_coherence: 各轮 query embedding 的语义一致性
    raise NotImplementedError(
        "cross_turn_consistency (API) 尚未实现。"
        "需要多轮 API 数据集（如 tau-bench）实际接入后设计具体检测逻辑。"
    )
