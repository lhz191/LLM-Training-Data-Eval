#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮 Text-GUI Agent 数据加载器

将 WebLINX 等多轮数据集加载为 Session 对象。
每个 Session 按 instructor utterance 切分为多轮 Record。

使用方式:
    from mt_text_gui_agent_eval.loaders import MultiTurnWebLINXLoader

    loader = MultiTurnWebLINXLoader('/path/to/weblinx/chat_data/data/chat')
    for session in loader.iterate():
        print(session)  # Session(id='mt_weblinx_0', rounds=4)
"""

import gzip
import json
import os
import re
import sys
import importlib.util
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator, List, Dict, Optional, Tuple, Any
from urllib.parse import urlparse

_this_dir = os.path.dirname(os.path.abspath(__file__))
_agent_data_dir = os.path.dirname(_this_dir)
_st_dir = os.path.join(_agent_data_dir, 'text_gui_agent_eval')

if _st_dir not in sys.path:
    sys.path.insert(0, _st_dir)

_st_dt_path = os.path.join(_st_dir, 'data_types.py')
_spec = importlib.util.spec_from_file_location('st_data_types', _st_dt_path)
_st_dt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_st_dt)
Record = _st_dt.Record
Action = _st_dt.Action

_mt_dt_path = os.path.join(_this_dir, 'data_types.py')
_spec2 = importlib.util.spec_from_file_location('mt_data_types', _mt_dt_path)
_mt_dt = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mt_dt)
Session = _mt_dt.Session


# =============================================================================
# Base Loader
# =============================================================================

class BaseMultiTurnLoader(ABC):
    """多轮数据集加载器基类，与单轮 BaseLoader 接口对齐"""

    def __init__(self, data_path: str):
        self.data_path = data_path

    @abstractmethod
    def load(self) -> List[Dict]:
        """加载原始数据，子类需实现"""
        pass

    @abstractmethod
    def parse_all(self, show_progress: bool = True) -> List[Session]:
        """解析所有记录为 Session 列表，子类需实现"""
        pass

    @abstractmethod
    def iterate(self, show_progress: bool = True) -> Iterator[Session]:
        """迭代返回 Session，子类需实现"""
        pass

    @abstractmethod
    def parse_session(self, raw_data: Any, idx: int = 0) -> Optional[Session]:
        """解析单条记录为 Session，子类需实现"""
        pass


# =============================================================================
# 辅助函数
# =============================================================================

def _parse_utterances(utt_str: str) -> List[Tuple[str, str]]:
    """
    从 WebLINX utterances 字符串中提取 (timestamp, message) 列表。

    格式: "[-00:08] Hello [00:33] Open momondo.in and login... [01:43] I am searching..."
    返回: [("-00:08", "Hello"), ("00:33", "Open momondo.in..."), ...]
    """
    if not utt_str or 'N o   i n s t r u c t o r' in utt_str:
        return []

    parts = re.split(r'\[([^\]]+)\]\s*', utt_str)
    results = []
    i = 1
    while i < len(parts) - 1:
        ts = parts[i].strip()
        msg = parts[i + 1].strip().rstrip(';').strip()
        if msg:
            results.append((ts, msg))
        i += 2
    return results


def _parse_action_string(action_str: str) -> Tuple[str, str]:
    """解析 WebLINX action 字符串为 (type, value)"""
    if not action_str:
        return ('unknown', '')

    match = re.match(r'(\w+)\(', action_str)
    if not match:
        return (action_str, '')

    action_type = match.group(1)

    if action_type == 'say':
        m = re.search(r'utterance="(.*?)"\)', action_str, re.DOTALL)
        return ('say', m.group(1) if m else '')
    elif action_type == 'text_input':
        m = re.search(r'text="(.*?)"', action_str, re.DOTALL)
        return ('text_input', m.group(1) if m else '')
    elif action_type == 'load':
        m = re.search(r'url="(.*?)"', action_str, re.DOTALL)
        return ('load', m.group(1) if m else '')
    elif action_type in ('click', 'hover'):
        m = re.search(r'uid="(.*?)"', action_str)
        return (action_type, m.group(1) if m else '')
    elif action_type == 'scroll':
        m = re.search(r'x=(\d+),\s*y=(\d+)', action_str)
        return ('scroll', f"{m.group(1)},{m.group(2)}" if m else '')
    elif action_type == 'change':
        m = re.search(r'value="(.*?)"', action_str, re.DOTALL)
        return ('change', m.group(1) if m else '')
    elif action_type == 'submit':
        m = re.search(r'uid="(.*?)"', action_str)
        return ('submit', m.group(1) if m else '')
    else:
        return (action_type, '')


def _extract_target(action_str: str) -> Optional[str]:
    """从 action 字符串中提取 target uid"""
    m = re.search(r'uid="([^"]+)"', action_str)
    return m.group(1) if m else None


# =============================================================================
# WebLINX Multi-Turn Loader
# =============================================================================

class MultiTurnWebLINXLoader(BaseMultiTurnLoader):
    """
    多轮 WebLINX 加载器

    按 instructor utterance 切分每个 demo 为多轮 Session。
    每当 instructor 说了新的话，就开启新一轮 Record。

    数据特点：
    - 数据按 action 分割，需要按 demo 聚合
    - utterances 字段包含累积的 instructor 对话历史
    - say(speaker="navigator", ...) 是 agent 说的话
    - instructor 的新指令通过 utterances 字段增长来检测
    - train.json.gz 格式（gzip 压缩的 JSONL）

    使用方式:
        loader = MultiTurnWebLINXLoader('/path/to/chat', split='train')
        for session in loader.iterate():
            print(session)

    数据路径：
    - /mnt/petrelfs/liuhaoze/datasets/Agent_Data/weblinx/chat_data/data/chat/train.json.gz
    """

    def __init__(self, data_dir: str, split: str = 'train'):
        filepath = os.path.join(data_dir, f'{split}.json.gz')
        super().__init__(filepath)
        self.data_dir = data_dir
        self.split = split
        self.data: List[Dict] = []
        self.demos: Dict[str, List[Dict]] = {}

    def load(self) -> List[Dict]:
        """加载原始数据并按 demo 聚合"""
        filepath = os.path.join(self.data_dir, f'{self.split}.json.gz')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"WebLINX 数据文件不存在: {filepath}")

        print(f"Loading WebLINX MT ({self.split}): {filepath}")

        self.data = []
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

        demos = defaultdict(list)
        for record in self.data:
            demos[record['demo']].append(record)

        for demo_id in demos:
            demos[demo_id].sort(key=lambda x: x['turn'])

        self.demos = dict(demos)
        print(f"Loaded {len(self.data):,} actions from {len(self.demos):,} demos")
        return self.data

    def parse_all(self, show_progress: bool = True) -> List[Session]:
        """解析所有 demo 为 Session 列表"""
        if not self.demos:
            self.load()

        sessions = []
        demo_ids = list(self.demos.keys())

        if show_progress:
            from tqdm import tqdm
            demo_ids = tqdm(demo_ids, desc="Parsing sessions")

        for idx, demo_id in enumerate(demo_ids):
            session = self.parse_session(demo_id, idx)
            if session:
                sessions.append(session)

        print(f"Parsed {len(sessions):,} sessions")
        return sessions

    def iterate(self, show_progress: bool = True) -> Iterator[Session]:
        """迭代返回 Session"""
        if not self.demos:
            self.load()

        demo_ids = list(self.demos.keys())

        if show_progress:
            from tqdm import tqdm
            demo_ids = tqdm(demo_ids, desc="Iterating sessions")

        for idx, demo_id in enumerate(demo_ids):
            session = self.parse_session(demo_id, idx)
            if session:
                yield session

    def parse_session(self, demo_id: str, idx: int = 0) -> Optional[Session]:
        """
        解析单个 demo 为 Session

        Args:
            demo_id: demo ID
            idx: Session 索引

        Returns:
            Session 或 None
        """
        raw_actions = self.demos.get(demo_id)
        if not raw_actions:
            return None

        try:
            website = self._extract_website(raw_actions)
            round_splits = self._split_into_rounds(raw_actions)

            records = []
            for r_idx, (instruction, raw_list) in enumerate(round_splits):
                actions = []
                for a_idx, raw in enumerate(raw_list):
                    action = self._parse_action(raw, a_idx)
                    if action:
                        actions.append(action)

                if not actions:
                    continue

                records.append(Record(
                    actions=actions,
                    sample_id=f"mt_weblinx_{idx}_r{r_idx}",
                    instruction=instruction or None,
                    website=website,
                ))

            if not records:
                return None

            full_utt = self._extract_full_utterances(raw_actions)

            return Session(
                rounds=records,
                session_id=f"mt_weblinx_{idx}",
                metadata={
                    'demo_id': demo_id,
                    'full_utterances': full_utt,
                    'website': website,
                },
            )

        except Exception as e:
            print(f"Warning: failed to parse demo {demo_id}: {e}")
            return None

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _extract_website(self, raw_actions: List[Dict]) -> Optional[str]:
        """从 action_history 中提取主网站域名"""
        for raw in raw_actions:
            ah = raw.get('action_history', '') or ''
            m = re.search(r'load\(url="([^"]+)"\)', ah)
            if m:
                try:
                    domain = urlparse(m.group(1)).netloc
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    return domain
                except Exception:
                    pass
        return None

    def _extract_full_utterances(self, raw_actions: List[Dict]) -> str:
        """提取最完整的 utterances（累积的对话历史）"""
        full_utterances = ""
        for raw in raw_actions:
            u = raw.get('utterances', '') or ''
            if u and 'N o   i n s t r u c t o r' not in u and len(u) > len(full_utterances):
                full_utterances = u
        return full_utterances

    def _split_into_rounds(self, raw_actions: List[Dict]) -> List[Tuple[str, List[Dict]]]:
        """
        按 instructor utterance 变化切分 actions 为多轮。

        每当 utterances 字段中出现新的 instructor 消息，
        就开启新一轮。

        Returns:
            [(instruction, [raw_actions]), ...] 列表
        """
        rounds: List[Tuple[str, List[Dict]]] = []
        prev_msg_count = 0
        current_instruction = ""
        current_actions: List[Dict] = []

        for raw in raw_actions:
            utt = raw.get('utterances', '') or ''
            messages = _parse_utterances(utt)
            msg_count = len(messages)

            if msg_count > prev_msg_count and messages:
                new_msg = messages[-1][1]

                if current_actions:
                    rounds.append((current_instruction, current_actions))
                    current_actions = []

                current_instruction = new_msg
                prev_msg_count = msg_count

            current_actions.append(raw)

        if current_actions:
            rounds.append((current_instruction, current_actions))

        return rounds

    def _parse_action(self, raw_action: Dict, action_idx: int) -> Optional[Action]:
        """解析单个 WebLINX action"""
        try:
            action_str = raw_action.get('action', '')
            action_type, action_value = _parse_action_string(action_str)
            target = _extract_target(action_str)

            return Action(
                action_idx=action_idx,
                action_type=action_type,
                action_value=action_value,
                action_repr=action_str,
                cleaned_html=raw_action.get('clean_html', '') or '',
                target_element=target,
                candidates=raw_action.get('candidates', '') or '',
                metadata={
                    'turn': raw_action.get('turn', 0),
                    'viewport': raw_action.get('viewport', ''),
                },
            )

        except Exception as e:
            print(f"Warning: failed to parse action {action_idx}: {e}")
            return None


# =============================================================================
# 便捷函数
# =============================================================================

def load_weblinx_multiturn(
    data_dir: str,
    split: str = 'train',
    show_progress: bool = True,
) -> List[Session]:
    """便捷函数：加载 WebLINX 数据集为多轮 Session 列表"""
    loader = MultiTurnWebLINXLoader(data_dir, split)
    return loader.parse_all(show_progress)


__all__ = [
    'BaseMultiTurnLoader',
    'MultiTurnWebLINXLoader',
    'load_weblinx_multiturn',
]
