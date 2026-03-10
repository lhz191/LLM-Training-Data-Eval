#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context Utilization -- 多轮 API Agent 专有指标

评估多轮 API 会话中 agent 对历史上下文的利用程度。

=== 检测维度 ===
1. 返回值复用率：前轮 API 返回的信息是否在后续轮次被引用
2. 参数传递成功率：前轮输出作为后轮输入的链式调用是否正确衔接
3. 上下文遗忘率：agent 是否"忘记"前轮约束（如用户偏好）
4. 累积效率：后续轮次是否因复用上下文而减少冗余调用
"""

import os
import sys
from typing import Iterator, Dict, Any, Optional

_this_dir = os.path.dirname(os.path.abspath(__file__))
_mt_dir = os.path.dirname(_this_dir)

if _mt_dir not in sys.path:
    sys.path.insert(0, _mt_dir)

from data_types import Session  # noqa: E402


def compute_context_utilization(
    session_iterator: Iterator[Session],
    dataset_name: str,
    output_file: Optional[str] = None,
    max_sessions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    计算多轮 API 会话的上下文利用率

    Returns:
        {
            'dataset': str,
            'total_sessions': int,
            'return_value_reuse_rate': float,
            'param_chaining_success_rate': float,
            'context_forgetting_rate': float,
            'cumulative_efficiency_score': float,
            'overall_utilization_score': float,
        }
    """
    # TODO: 实现
    # 1. return_value_reuse: 检查前轮 API response 中的值是否出现在后轮的参数中
    # 2. param_chaining: 检测 A 轮返回的 order_id 是否传入 B 轮的 get_order_status
    # 3. context_forgetting: 前轮确定的约束（如 city="北京"）后续是否违反
    # 4. cumulative_efficiency: 后续轮次的 API 调用数量趋势
    raise NotImplementedError(
        "context_utilization (API) 尚未实现。"
        "需要多轮 API 数据集实际接入后设计具体检测逻辑。"
    )
