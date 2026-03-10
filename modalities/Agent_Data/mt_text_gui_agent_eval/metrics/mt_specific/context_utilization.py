#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上下文利用率 (Context Utilization) — 多轮专有指标

评估多轮会话中 agent 对历史上下文的利用程度。

=== 核心问题 ===
多轮会话中，前几轮的交互提供了重要的上下文信息。
一个好的 agent 应该利用这些信息，而不是每轮"从零开始"。
这对评估训练数据质量很重要——如果数据中 agent 不利用上下文，
用这些数据训练出的模型也不会利用上下文。

=== 检测维度 ===

1. 信息复用率 (Information Reuse)
   前轮产生的信息（搜索结果、选择的选项、填写的内容）
   是否在后续轮次被正确引用/使用。
   - 例：第1轮搜索了航班列表，第2轮选择其中一个，
         而非重新搜索。

2. 指代消解成功率 (Reference Resolution)
   后续轮次中的指代/省略是否被正确解析。
   - 例：第1轮 "搜索 iPhone 16"，第2轮 "看看它的评价"，
         agent 应知道"它"指 iPhone 16。

3. 上下文遗忘率 (Context Forgetting)
   agent 在后续轮次是否"忘记"了之前轮次的关键信息。
   - 例：第1轮用户说 "我要明天的"，第3轮 agent 选了今天的。

4. 累积效率 (Cumulative Efficiency)
   多轮交互是否逐轮变得更高效（动作数递减），
   还是每轮都像第一次一样冗长。

=== 使用方式 ===
    from mt_text_gui_agent_eval.data_types import Session
    from mt_text_gui_agent_eval.metrics.context_utilization import compute_context_utilization

    results = compute_context_utilization(
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

from data_types import Session  # noqa: E402


def compute_context_utilization(
    session_iterator: Iterator[Session],
    dataset_name: str,
    output_file: Optional[str] = None,
    max_sessions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    计算多轮会话的上下文利用率

    Args:
        session_iterator: Session 迭代器
        dataset_name: 数据集名称
        output_file: 输出文件路径
        max_sessions: 最大会话数

    Returns:
        {
            'dataset': str,
            'total_sessions': int,
            'information_reuse_rate': float,        # 信息复用率 [0, 1]
            'reference_resolution_rate': float,     # 指代消解成功率 [0, 1]
            'context_forgetting_rate': float,       # 上下文遗忘率 [0, 1]（越低越好）
            'cumulative_efficiency_score': float,   # 累积效率 [0, 1]
            'overall_utilization_score': float,
            'per_session_results': [...]
        }
    """
    # TODO: 实现上下文利用率检测
    # 思路：
    # 1. information_reuse: 提取每轮的"产出"（搜索结果、选中项等），
    #    检查后续轮次的 action 是否引用了这些产出
    # 2. reference_resolution: 检测 instruction 中的指代词（它、这个、那个），
    #    判断 agent 的 action 是否指向了正确的实体
    # 3. context_forgetting: 提取前轮的约束条件（日期、地点等关键词），
    #    检查后续轮次是否违反了这些约束
    # 4. cumulative_efficiency: 计算每轮的 action 数量趋势，
    #    正常情况下后续轮次应该更简短（因为可以复用之前的状态）
    raise NotImplementedError(
        "context_utilization 尚未实现。"
        "需要多轮数据集实际接入后，根据数据特点设计具体检测逻辑。"
    )
