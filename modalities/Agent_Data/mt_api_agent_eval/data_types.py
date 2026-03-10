#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮 API Agent 数据类型

Session 定义在此处。APIAgentSample / ToolDefinition / APICall / Parameter
从单轮模块导入复用。
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

_this_dir = os.path.dirname(os.path.abspath(__file__))
_agent_data_dir = os.path.dirname(_this_dir)
sys.path.insert(0, os.path.join(_agent_data_dir, 'api_agent_eval'))

from data_types import APIAgentSample, ToolDefinition, APICall, Parameter  # noqa: E402


@dataclass
class Session:
    """
    多轮会话：用户与 API Agent 的多轮交互

    每个 round 是一个 APIAgentSample（一次查询 + 工具调用序列）。
    rounds 按时间顺序排列。

    示例（多轮客服场景，如 tau-bench）：
        session_id: "session_001"
        rounds: [
            APIAgentSample(query="帮我查一下订单状态", tools=[...], api_calls=[...]),
            APIAgentSample(query="把收货地址改成北京", tools=[...], api_calls=[...]),
            APIAgentSample(query="再帮我取消那个订单", tools=[...], api_calls=[...]),
        ]

    适用数据集：
    - APIGen-MT（合成多轮 tool-use 对话）
    - tau-bench（多轮客服场景）
    - ToolTalk（多轮工具调用对话）
    - 合成多轮数据
    """

    rounds: List[APIAgentSample]

    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        id_str = f"id='{self.session_id}', " if self.session_id else ""
        return f"Session({id_str}rounds={len(self.rounds)})"

    @property
    def is_multi_turn(self) -> bool:
        return len(self.rounds) > 1

    @property
    def total_api_calls(self) -> int:
        return sum(len(r.api_calls) for r in self.rounds)

    def get_all_queries(self) -> List[str]:
        return [r.query for r in self.rounds]

    def get_all_tools(self) -> List[ToolDefinition]:
        """获取所有轮次中出现的工具（去重按 name）"""
        seen = set()
        tools = []
        for r in self.rounds:
            for t in r.tools:
                if t.name not in seen:
                    seen.add(t.name)
                    tools.append(t)
        return tools

    def get_all_api_calls(self) -> List[APICall]:
        calls = []
        for r in self.rounds:
            calls.extend(r.api_calls)
        return calls

    def get_all_samples(self) -> List[APIAgentSample]:
        return list(self.rounds)


__all__ = ['Session', 'APIAgentSample', 'ToolDefinition', 'APICall', 'Parameter']
