#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-based GUI Agent Evaluation - 数据类型定义

统一的 Text-based GUI Agent 数据格式，支持：
- Mind2Web：真实网站，HTML + 候选元素
- WebShop：模拟电商，文本状态 + 可选动作
- WebLINX：真实网站，HTML + 多轮对话

设计原则：
1. 字段定义参照真实数据条目
2. 提供统一接口，不同数据集通过 Loader 转换为统一格式
3. candidates 保持原始格式，不强制统一（各数据集差异太大）
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


# =============================================================================
# 动作定义
# =============================================================================

@dataclass
class Action:
    """
    单个动作
    
    支持 Mind2Web、WebShop、WebLINX 三种数据集格式。
    
    Mind2Web 示例：
        action_idx: 0
        action_type: "click"
        action_value: ""  # click 无附加值
        action_repr: "[button] Search -> CLICK"
        cleaned_html: "<html>..."
        target_element: {"tag": "button", "backend_node_id": "123", ...}
        candidates: [pos_candidates + neg_candidates]
    
    Mind2Web TYPE 示例：
        action_idx: 2
        action_type: "type"
        action_value: "Prometheus"  # 输入的文字
        action_repr: "[searchbox] Search -> TYPE: Prometheus"
        ...
    
    WebShop 示例：
        action_idx: 0
        action_type: "search"
        action_value: "black ottoman"  # 搜索内容
        action_repr: "search[black ottoman]"
        cleaned_html: "Instruction: ... [button] back to search [button] next > ..."
        target_element: "search[black ottoman]"
        candidates: ["click[back to search]", "click[next >]", ...]
    
    WebLINX 示例：
        action_idx: 0
        action_type: "say"
        action_value: "Hello, I'll help you..."  # 说的话
        action_repr: "say(utterance=\"Hello, I'll help you...\")"
        cleaned_html: "<html>..."
        target_element: None  # say 动作无目标元素
        candidates: []
    """
    
    # === 必需字段 ===
    action_idx: int                         # 动作序号 (第几个动作，从 0 开始)
    action_type: str                        # 动作类型 (click/type/select/scroll/say/search/hover/etc.)
    action_value: str = ""                  # 动作附加值 (可为空)
                                            # Mind2Web: operation.value (如 type 时为 "Prometheus")
                                            # WebShop: search[xxx] 中的 xxx
                                            # WebLINX: say(utterance="xxx") 中的 xxx
                                            # click 等无附加值的动作为 ""
    action_repr: str = ""                   # 动作完整表示
                                            # Mind2Web: "[button] Search -> CLICK"
                                            # WebShop: "click[item - xxx]" 或 "search[xxx]"
                                            # WebLINX: "click(uid=\"xxx\")" 或 "say(utterance=\"xxx\")"
    cleaned_html: str = ""                  # 清理后的 HTML / 文本状态 (Text Agent 核心输入)
                                            # Mind2Web: cleaned_html
                                            # WebShop: state (文本表示)
                                            # WebLINX: clean_html
                                            # 也可以是 DOM / Accessibility Tree 等文本表示
    
    # === 可选字段 ===
    raw_html: Optional[str] = None          # 原始 HTML (Mind2Web 有，其他可能无)
    screenshot: Optional[str] = None        # 截图路径 (Text+Vision 模式)
    target_element: Any = None              # 目标元素 (正确答案，格式因数据集而异)
                                            # Mind2Web: Dict (pos_candidates[0])
                                            # WebShop: str (动作本身)
                                            # WebLINX: str (uid)
                                            # say/scroll 等动作可能为 None
    candidates: List[Any] = field(default_factory=list)
                                            # 所有候选 (包含正确和错误，格式因数据集而异)
                                            # Mind2Web: List[Dict] (pos + neg candidates)
                                            # WebShop: List[str] (available_actions)
                                            # WebLINX: str (一个大字符串，需要解析)
                                            # say/scroll 等动作可能为空列表
    metadata: Dict[str, Any] = field(default_factory=dict)
                                            # 其他数据集特有的字段
                                            # Mind2Web: {"action_uid": "xxx"} (用于找 HTML 文件)
                                            # WebLINX: {"utterances": "...", "action_history": "...", "turn": 6}
                                            # WebShop: {}
    
    def __repr__(self):
        return f"Action(idx={self.action_idx}, type='{self.action_type}', repr='{self.action_repr[:30]}...')"


# =============================================================================
# 记录定义 (一个完整的任务)
# =============================================================================

@dataclass 
class Record:
    """
    一个完整的任务记录
    
    一个 Record 包含一个任务目标和完成该任务的动作序列。
    原生 ID 存储在 metadata 中。
    
    Mind2Web 示例：
        sample_id: "mind2web_0"
        actions: [Action, Action, ...]
        instruction: "Search for flights from NYC to LA"
        website: "google.com"
        metadata: {"annotation_id": "4e56a7b8-..."}
    
    WebShop 示例：
        sample_id: "webshop_0"
        actions: [Action, Action, ...]
        instruction: "I need a black ottoman for my living room..."
        website: "webshop"
        metadata: {"goal_idx": 123, "reward": 1.0}
    
    WebLINX 示例：
        sample_id: "weblinx_0"
        actions: [Action, Action, ...]
        instruction: None  # 无汇总的 instruction
        website: "momondo.in"
        metadata: {"demo_id": "aaabtsd", "full_utterances": "..."}
    """
    
    # === 必需字段 ===
    actions: List[Action]                   # 动作序列
    
    # === 元信息 ===
    sample_id: Optional[str] = None         # 样本唯一标识 (Loader 生成，如 "mind2web_0")
    
    # === 可选字段 ===
    instruction: Optional[str] = None       # 任务目标/指令 (主要供人阅读，评估时不一定用到)
                                            # Mind2Web: confirmed_task (原生字段)
                                            # WebShop: states[0] 中的 instruction 部分 (需解析)
                                            # WebLINX: 无汇总的，若需要，从 utterances 提取 (Loader 处理，无原生字段)
    website: Optional[str] = None           # 网站信息
                                            # Mind2Web: website
                                            # WebShop: "webshop" (固定)
                                            # WebLINX: 从 URL 提取
    metadata: Dict[str, Any] = field(default_factory=dict)
                                            # 其他元信息
                                            # Mind2Web: {"annotation_id": "xxx", "domain": "...", "subdomain": "..."}
                                            # WebShop: {"goal_idx": 123, "reward": 1.0}
                                            # WebLINX: {"demo_id": "aaabtsd", "full_utterances": "..."}
    
    def __repr__(self):
        id_str = f"id='{self.sample_id}', " if self.sample_id else ""
        if self.instruction:
            instr_preview = self.instruction[:50] + "..." if len(self.instruction) > 50 else self.instruction
            return f"Record({id_str}instruction='{instr_preview}', actions={len(self.actions)})"
        else:
            return f"Record({id_str}actions={len(self.actions)})"
    
    def get_action_types(self) -> List[str]:
        """获取所有动作类型"""
        return [a.action_type for a in self.actions]
    
    def get_action_reprs(self) -> List[str]:
        """获取所有动作表示"""
        return [a.action_repr for a in self.actions]
