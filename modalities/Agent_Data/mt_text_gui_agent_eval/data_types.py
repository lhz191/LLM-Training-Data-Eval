#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮 Text-GUI Agent 数据类型

Session 定义在此处。Record / Action 从单轮模块导入复用。
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

_this_dir = os.path.dirname(os.path.abspath(__file__))
_agent_data_dir = os.path.dirname(_this_dir)
sys.path.insert(0, os.path.join(_agent_data_dir, 'text_gui_agent_eval'))

from data_types import Record, Action  # noqa: E402


@dataclass
class Session:
    """
    多轮会话：用户与 Web Agent 的多轮交互

    每个 round 是一个 Record（一次任务指令 + 动作序列）。
    rounds 按时间顺序排列。

    示例（多轮网页操作）：
        session_id: "session_001"
        rounds: [
            Record(instruction="搜索北京到上海的机票", actions=[...]),
            Record(instruction="改成高铁，明天出发", actions=[...]),
            Record(instruction="选最便宜的那个", actions=[...]),
        ]

    适用数据集：
    - WebLINX（天然多轮对话式网页导航）
    - MT-Mind2Web（Mind2Web 多轮扩展版）
    - 合成多轮数据
    """

    rounds: List[Record]

    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        id_str = f"id='{self.session_id}', " if self.session_id else ""
        return f"Session({id_str}rounds={len(self.rounds)})"

    @property
    def is_multi_turn(self) -> bool:
        return len(self.rounds) > 1

    @property
    def total_actions(self) -> int:
        return sum(len(r.actions) for r in self.rounds)

    def get_all_instructions(self) -> List[Optional[str]]:
        return [r.instruction for r in self.rounds]

    def get_all_actions(self) -> List[Action]:
        actions = []
        for r in self.rounds:
            actions.extend(r.actions)
        return actions

    def get_all_records(self) -> List[Record]:
        return list(self.rounds)


__all__ = ['Session', 'Record', 'Action']
