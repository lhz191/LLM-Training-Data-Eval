#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-based GUI Agent Data Loaders

数据加载器，将不同格式的数据集转换为统一的 Record 格式。

支持：
- Mind2Web：真实网站，HTML + 候选元素
- Multimodal Mind2Web：Mind2Web + 截图
- WebShop：模拟电商，文本状态 + 可选动作 (TODO)
- WebLINX：真实网站，HTML + 多轮对话 (TODO)
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from collections import defaultdict
from tqdm import tqdm

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from data_types import Action, Record


# =============================================================================
# Base Loader
# =============================================================================

class BaseLoader:
    """数据集加载器基类"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load(self) -> List[Dict]:
        """加载原始数据，子类需实现"""
        raise NotImplementedError
    
    def parse_all(self, show_progress: bool = True) -> List[Record]:
        """解析所有记录为 Record 列表，子类需实现"""
        raise NotImplementedError
    
    def iterate(self, show_progress: bool = True) -> Iterator[Record]:
        """迭代返回 Record，子类需实现"""
        raise NotImplementedError
    
    def parse_record(self, raw_record: Dict, idx: int = 0) -> Optional[Record]:
        """解析单条记录，子类需实现"""
        raise NotImplementedError


# =============================================================================
# Mind2Web Loader
# =============================================================================

class Mind2WebLoader(BaseLoader):
    """
    Mind2Web 数据集加载器
    
    将 Mind2Web 的多步骤任务格式转换为统一的 Record。
    
    Mind2Web 原始数据格式 (JSON)：
    {
        "website": "ign",
        "domain": "Entertainment",
        "subdomain": "Game",
        "annotation_id": "39b037ac-0a11-4b05-8919-b4f9863fd0cd",
        "confirmed_task": "Show review of Prometheus movie.",
        "action_reprs": ["[path] -> CLICK", "[tab] MOVIES -> CLICK", ...],
        "actions": [
            {
                "action_uid": "f4a3db2b-...",
                "raw_html": "<!DOCTYPE html>...",
                "cleaned_html": "<html>...</html>",
                "operation": {"original_op": "CLICK", "value": "", "op": "CLICK"},
                "pos_candidates": [{"tag": "svg", "backend_node_id": "486", ...}],
                "neg_candidates": [{"tag": "html", "backend_node_id": "127", ...}]
            },
            ...
        ]
    }
    
    转换为 Record:
        sample_id: "mind2web_0"
        actions: [Action, Action, ...]
        instruction: "Show review of Prometheus movie."
        website: "ign"
        metadata: {"annotation_id": "...", "domain": "...", "subdomain": "...", "action_reprs": [...]}
    """
    
    def __init__(self, data_path: str):
        """
        初始化 Mind2Web 加载器
        
        Args:
            data_path: 数据路径，支持三种格式：
                1. 目录路径 (如 .../Mind2Web/data) - 自动查找或生成 train_all.json
                2. JSON 文件路径 (如 train_all.json)
                3. 包含多个 train_*.json 的目录 (如 .../Mind2Web/data/train)
        """
        super().__init__(data_path)
        self.data: List[Dict] = []
    
    # =========================================================================
    # 数据加载和解析方法
    # =========================================================================
    
    def load(self) -> List[Dict]:
        """
        加载原始 JSON 数据
        
        支持：
        1. 直接加载 train_all.json
        2. 如果传入目录，查找 train_all.json 或合并 train/*.json
        """
        data_path = Path(self.data_path)
        
        # 情况1: 直接传入 JSON 文件
        if data_path.is_file() and data_path.suffix == '.json':
            print(f"📂 Loading Mind2Web: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✅ Loaded {len(self.data):,} records")
            return self.data
        
        # 情况2: 传入目录
        if data_path.is_dir():
            # 检查是否有 train_all.json
            train_all_json = data_path / 'train_all.json'
            if train_all_json.exists():
                print(f"📂 Loading Mind2Web: {train_all_json}")
                with open(train_all_json, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"✅ Loaded {len(self.data):,} records")
                return self.data
            
            # 查找 train 子目录
            train_dir = data_path / 'train'
            if not train_dir.exists():
                train_dir = data_path  # 可能直接传入 train 目录
            
            # 查找所有 train_*.json 文件
            json_files = sorted(train_dir.glob('train_*.json'), 
                               key=lambda x: int(x.stem.split('_')[1]))
            
            if not json_files:
                raise FileNotFoundError(f"❌ No train_*.json files found in {train_dir}")
            
            # 合并所有 JSON 文件
            print(f"📂 Merging {len(json_files)} train files from {train_dir}")
            self.data = []
            for json_file in tqdm(json_files, desc="Loading JSON files"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.data.extend(json.load(f))
            
            # 保存合并后的文件
            output_path = data_path / 'train_all.json' if data_path != train_dir else data_path.parent / 'train_all.json'
            print(f"💾 Saving merged file to {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False)
            
            print(f"✅ Loaded and merged {len(self.data):,} records")
            return self.data
        
        raise FileNotFoundError(f"❌ Invalid path: {self.data_path}")
    
    def parse_all(self, show_progress: bool = True) -> List[Record]:
        """
        解析所有记录为 Record
        
        Args:
            show_progress: 是否显示进度条
        
        Returns:
            Record 列表
        """
        if not self.data:
            self.load()
        
        records = []
        iterator = tqdm(self.data, desc="Parsing Mind2Web") if show_progress else self.data
        
        for idx, raw_record in enumerate(iterator):
            record = self.parse_record(raw_record, idx)
            if record:
                records.append(record)
        
        print(f"✅ Parsed {len(records):,} records")
        return records
    
    def iterate(self, show_progress: bool = True) -> Iterator[Record]:
        """
        迭代返回 Record
        
        注意：Mind2Web 是 JSON 格式，需要一次性加载到内存。
        
        Args:
            show_progress: 是否显示进度条
        
        Yields:
            Record
        """
        if not self.data:
            self.load()
        
        iterator = tqdm(self.data, desc="Parsing Mind2Web") if show_progress else self.data
        
        for idx, raw_record in enumerate(iterator):
            record = self.parse_record(raw_record, idx)
            if record:
                yield record
    
    def parse_record(self, raw_record: Dict, idx: int = 0) -> Optional[Record]:
        """
        解析单条 Mind2Web 记录
        
        Args:
            raw_record: 原始记录
            idx: 记录索引
        
        Returns:
            Record 或 None（解析失败时）
        """
        try:
            # === 1. 解析 Actions ===
            actions = []
            raw_actions = raw_record.get('actions', [])
            action_reprs = raw_record.get('action_reprs', [])
            
            for i, raw_action in enumerate(raw_actions):
                action = self._parse_action(raw_action, i, action_reprs)
                if action:
                    actions.append(action)
            
            if not actions:
                return None
            
            # === 2. 构建 Record ===
            record = Record(
                actions=actions,
                sample_id=f"mind2web_{idx}",
                instruction=raw_record.get('confirmed_task', ''),
                website=raw_record.get('website', ''),
                metadata={
                    'annotation_id': raw_record.get('annotation_id', ''),
                    'domain': raw_record.get('domain', ''),
                    'subdomain': raw_record.get('subdomain', ''),
                    'action_reprs': action_reprs,
                }
            )
            
            return record
            
        except Exception as e:
            print(f"⚠️ Failed to parse record {idx}: {e}")
            return None
    
    def _parse_action(self, raw_action: Dict, idx: int, action_reprs: List[str]) -> Optional[Action]:
        """
        解析单个 Mind2Web action
        
        Args:
            raw_action: 原始 action 数据
            idx: action 索引
            action_reprs: action_reprs 列表（用于获取 action_repr）
        
        Returns:
            Action 或 None
        """
        try:
            # 获取 operation
            operation = raw_action.get('operation', {})
            action_type = operation.get('op', '').lower()
            action_value = operation.get('value', '')
            
            # 获取 action_repr
            action_repr = action_reprs[idx] if idx < len(action_reprs) else ''
            
            # 获取 pos_candidates 和 neg_candidates
            pos_candidates = raw_action.get('pos_candidates', [])
            neg_candidates = raw_action.get('neg_candidates', [])
            
            # 目标元素是第一个 pos_candidate
            target_element = pos_candidates[0] if pos_candidates else None
            
            # 合并所有 candidates
            all_candidates = pos_candidates + neg_candidates
            
            action = Action(
                action_idx=idx,
                action_type=action_type,
                action_value=action_value,
                action_repr=action_repr,
                cleaned_html=raw_action.get('cleaned_html', ''),
                raw_html=raw_action.get('raw_html'),
                screenshot=None,  # Mind2Web 无截图，Multimodal Mind2Web 有
                target_element=target_element,
                candidates=all_candidates,
                metadata={
                    'action_uid': raw_action.get('action_uid', ''),
                    'operation': operation,
                }
            )
            
            return action
            
        except Exception as e:
            print(f"⚠️ Failed to parse action {idx}: {e}")
            return None


# =============================================================================
# Multimodal Mind2Web Loader
# =============================================================================

class MultimodalMind2WebLoader(BaseLoader):
    """
    Multimodal Mind2Web 数据集加载器
    
    和 Mind2Web 字段几乎相同，区别在于：
    1. 数据格式：Parquet（每行是一个 Action，需要按 annotation_id 聚合）
    2. 新增字段：screenshot, target_action_index, target_action_reprs
    
    聚合逻辑：
    - 同一个 annotation_id 的所有行 = 同一个 Record 的所有 actions
    - 按 target_action_index 排序
    - Record 级别信息（confirmed_task, action_reprs 等）从第一行取
    """
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        初始化 Multimodal Mind2Web 加载器
        
        Args:
            data_dir: 数据目录路径 (包含 parquet 文件)
            split: 数据集分割 ('train', 'test_task', 'test_website', 'test_domain')
        """
        super().__init__(data_dir)
        self.data_dir = Path(data_dir)
        self.split = split
        self.data: List[Dict] = []  # 聚合后的 record 数据
        
        if not HAS_PANDAS:
            raise ImportError("需要安装 pandas: pip install pandas pyarrow")
    
    def _find_parquet_files(self) -> List[Path]:
        """查找指定 split 的 parquet 文件"""
        pattern = f"{self.split}-*.parquet"
        files = sorted(self.data_dir.glob(pattern))
        return files
    
    def load(self) -> List[Dict]:
        """
        加载并聚合原始数据
        
        将分散的 Action 行按 annotation_id 聚合成 Record
        """
        parquet_files = self._find_parquet_files()
        if not parquet_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中找不到 {self.split}-*.parquet 文件")
        
        print(f"📂 Loading Multimodal Mind2Web ({self.split}): {len(parquet_files)} files")
        
        # 收集所有 action 行，按 annotation_id 分组
        ann_id_to_rows = defaultdict(list)
        
        for pq_file in tqdm(parquet_files, desc="Loading parquet files"):
            df = pd.read_parquet(pq_file)
            for _, row in df.iterrows():
                ann_id = row['annotation_id']
                ann_id_to_rows[ann_id].append(row.to_dict())
        
        # 聚合成 Record 格式
        self.data = []
        for ann_id, rows in ann_id_to_rows.items():
            # 按 target_action_index 排序（in-place）
            # 每个 action 都应该有 target_action_index，默认 0 以防万一
            rows.sort(key=lambda x: int(x.get('target_action_index', 0)))
            
            # 取第一行的 Record 级别信息
            first_row = rows[0]
            
            record_data = {
                'annotation_id': ann_id,
                'website': first_row.get('website', ''),
                'domain': first_row.get('domain', ''),
                'subdomain': first_row.get('subdomain', ''),
                'confirmed_task': first_row.get('confirmed_task', ''),
                'action_reprs': list(first_row.get('action_reprs', [])),
                'actions': rows,  # 所有 action 行
            }
            self.data.append(record_data)
        
        print(f"✅ Loaded {len(self.data):,} records (from {sum(len(r['actions']) for r in self.data):,} actions)")
        return self.data
    
    def parse_all(self, show_progress: bool = True) -> List[Record]:
        """
        解析所有记录为 Record
        
        Args:
            show_progress: 是否显示进度条
        
        Returns:
            Record 列表
        """
        if not self.data:
            self.load()
        
        records = []
        iterator = tqdm(self.data, desc="Parsing MM Mind2Web") if show_progress else self.data
        
        for idx, raw_record in enumerate(iterator):
            record = self.parse_record(raw_record, idx)
            if record:
                records.append(record)
        
        print(f"✅ Parsed {len(records):,} records")
        return records
    
    def iterate(self, show_progress: bool = True) -> Iterator[Record]:
        """迭代返回 Record"""
        if not self.data:
            self.load()
        
        iterator = tqdm(self.data, desc="Parsing MM Mind2Web") if show_progress else self.data
        
        for idx, raw_record in enumerate(iterator):
            record = self.parse_record(raw_record, idx)
            if record:
                yield record
    
    def parse_record(self, raw_record: Dict, idx: int = 0) -> Optional[Record]:
        """
        解析单条聚合后的记录
        
        Args:
            raw_record: 聚合后的记录（包含 actions 列表）
            idx: 记录索引
        
        Returns:
            Record 或 None
        """
        try:
            actions = []
            raw_actions = raw_record.get('actions', [])
            action_reprs = raw_record.get('action_reprs', [])
            
            for raw_action in raw_actions:
                action = self._parse_action(raw_action, action_reprs)
                if action:
                    actions.append(action)
            
            if not actions:
                return None
            
            record = Record(
                actions=actions,
                sample_id=f"mm_mind2web_{idx}",
                instruction=raw_record.get('confirmed_task', ''),
                website=raw_record.get('website', ''),
                metadata={
                    'annotation_id': raw_record.get('annotation_id', ''),
                    'domain': raw_record.get('domain', ''),
                    'subdomain': raw_record.get('subdomain', ''),
                    'action_reprs': action_reprs,
                }
            )
            
            return record
            
        except Exception as e:
            print(f"⚠️ Failed to parse record {idx}: {e}")
            return None
    
    def _parse_action(self, raw_action: Dict, action_reprs: List[str]) -> Optional[Action]:
        """
        解析单个 Multimodal Mind2Web action
        
        和 Mind2Web 的 _parse_action 几乎相同，区别在于：
        1. action_idx 从 target_action_index 获取
        2. 新增 screenshot 字段
        """
        try:
            # 获取 action_idx
            action_idx = int(raw_action.get('target_action_index', -1))
            
            # 获取 operation（可能是 JSON 字符串或 dict）
            operation = raw_action.get('operation', {})
            if isinstance(operation, str):
                try:
                    operation = json.loads(operation)
                except:
                    operation = {}
            
            action_type = operation.get('op', '').lower()
            action_value = operation.get('value', '')
            
            # 获取 action_repr
            action_repr = action_reprs[action_idx] if action_idx < len(action_reprs) else ''
            
            # 获取 pos_candidates 和 neg_candidates
            pos_candidates = list(raw_action.get('pos_candidates', []))
            neg_candidates = list(raw_action.get('neg_candidates', []))
            
            # 解析 candidates（可能是 JSON 字符串）
            def parse_candidate(cand):
                if isinstance(cand, str):
                    try:
                        return json.loads(cand)
                    except:
                        return {'raw': cand}
                return cand
            
            pos_candidates = [parse_candidate(c) for c in pos_candidates]
            neg_candidates = [parse_candidate(c) for c in neg_candidates]
            
            # 目标元素是第一个 pos_candidate
            target_element = pos_candidates[0] if pos_candidates else None
            
            # 合并所有 candidates
            all_candidates = pos_candidates + neg_candidates
            
            # 处理 screenshot - 直接存 bytes，可用 PIL 加载:
            # from PIL import Image; import io
            # img = Image.open(io.BytesIO(action.screenshot))
            screenshot_data = raw_action.get('screenshot')
            screenshot_bytes = None
            if isinstance(screenshot_data, dict):
                screenshot_bytes = screenshot_data.get('bytes')  # 直接存 bytes
            
            action = Action(
                action_idx=action_idx,
                action_type=action_type,
                action_value=action_value,
                action_repr=action_repr,
                cleaned_html=raw_action.get('cleaned_html', ''),
                raw_html=raw_action.get('raw_html'),
                screenshot=screenshot_bytes,  # bytes，可直接用 PIL 加载
                target_element=target_element,
                candidates=all_candidates,
                metadata={
                    'action_uid': raw_action.get('action_uid', ''),
                    'operation': operation,
                    'target_action_reprs': raw_action.get('target_action_reprs', ''),
                }
            )
            
            return action
            
        except Exception as e:
            print(f"⚠️ Failed to parse action: {e}")
            return None


# =============================================================================
# WebShop Loader
# =============================================================================

class WebShopLoader(BaseLoader):
    """
    WebShop 数据集加载器
    
    WebShop 原始数据格式 (JSONL，每行一个完整轨迹 = 一个 Record):
    {
        "actions": ["search[xxx]", "click[xxx]", ...],
        "states": ["Amazon Shopping Game\nInstruction: ...", ...],
        "available_actions": [[], ["click[back]", ...], ...],
        "actions_translate": ["search[xxx]", "click[item - xxx]", ...],
        "action_idxs": [-1, 0, 2, ...],
        "images": [0, 0, [512维向量], ...]
    }
    
    转换为 Record:
        sample_id: "webshop_0"
        actions: [Action, Action, ...]
        instruction: "i'm looking for living room furniture..."
        website: "webshop"
        metadata: {}
    
    Action 映射:
        action_idx: step 索引
        action_type: "search" 或 "click"
        action_value: search 的内容 或 click 的目标
        action_repr: actions[step] (原始动作表示)
        cleaned_html: states[step] (文本状态，作为 Text Agent 的输入)
        target_element: actions[step] (动作本身就是答案)
        candidates: available_actions[step]
    """
    
    def __init__(self, data_path: str):
        """
        初始化 WebShop 加载器
        
        Args:
            data_path: JSONL 数据文件路径 (如 il_trajs_finalized_images.jsonl)
        """
        super().__init__(data_path)
        self.data: List[Dict] = []
    
    def load(self) -> List[Dict]:
        """加载原始 JSONL 数据"""
        print(f"📂 Loading WebShop: {self.data_path}")
        
        self.data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"✅ Loaded {len(self.data):,} trajectories")
        return self.data
    
    def parse_all(self, show_progress: bool = True) -> List[Record]:
        """
        解析所有记录为 Record
        
        Args:
            show_progress: 是否显示进度条
        
        Returns:
            Record 列表
        """
        if not self.data:
            self.load()
        
        records = []
        iterator = tqdm(self.data, desc="Parsing WebShop") if show_progress else self.data
        
        for idx, raw_traj in enumerate(iterator):
            record = self.parse_record(raw_traj, idx)
            if record:
                records.append(record)
        
        print(f"✅ Parsed {len(records):,} records")
        return records
    
    def iterate(self, show_progress: bool = True) -> Iterator[Record]:
        """迭代返回 Record（逐行读取，节省内存）"""
        print(f"📂 Iterating WebShop: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            iterator = tqdm(enumerate(f), desc="Parsing WebShop") if show_progress else enumerate(f)
            
            for idx, line in iterator:
                if line.strip():
                    raw_traj = json.loads(line)
                    record = self.parse_record(raw_traj, idx)
                    if record:
                        yield record
    
    def parse_record(self, raw_traj: Dict, idx: int = 0) -> Optional[Record]:
        """
        解析单条轨迹为 Record
        
        Args:
            raw_traj: 原始轨迹数据
            idx: 轨迹索引
        
        Returns:
            Record 或 None
        """
        try:
            actions_raw = raw_traj.get('actions', [])
            states = raw_traj.get('states', [])
            available_actions = raw_traj.get('available_actions', [])
            actions_translate = raw_traj.get('actions_translate', [])
            action_idxs = raw_traj.get('action_idxs', [])
            images = raw_traj.get('images', [])
            
            if not actions_raw or not states:
                return None
            
            # 提取 instruction (从第一个 state 中提取)
            instruction = self._extract_instruction(states[0])
            
            # 解析每个 action
            actions = []
            for step_idx in range(len(actions_raw)):
                action = self._parse_action(
                    step_idx=step_idx,
                    action_raw=actions_raw[step_idx],
                    state=states[step_idx] if step_idx < len(states) else '',
                    avail_actions=available_actions[step_idx] if step_idx < len(available_actions) else [],
                    action_translate=actions_translate[step_idx] if step_idx < len(actions_translate) else '',
                    action_idx_in_avail=action_idxs[step_idx] if step_idx < len(action_idxs) else -1,
                    image=images[step_idx] if step_idx < len(images) else 0,
                )
                if action:
                    actions.append(action)
            
            if not actions:
                return None
            
            record = Record(
                actions=actions,
                sample_id=f"webshop_{idx}",
                instruction=instruction,
                website="webshop",
                metadata={}
            )
            
            return record
            
        except Exception as e:
            print(f"⚠️ Failed to parse trajectory {idx}: {e}")
            return None
    
    def _extract_instruction(self, state: str) -> str:
        """
        从 state 文本中提取 instruction
        
        格式: "Amazon Shopping Game\nInstruction: \nxxx\n[button]..."
        或: "Instruction:\nxxx\n[button]..."
        """
        lines = state.strip().split('\n')
        instruction = ''
        capture = False
        
        for line in lines:
            if 'Instruction:' in line:
                capture = True
                # 如果 Instruction: 后面有内容
                after = line.split('Instruction:')[-1].strip()
                if after:
                    instruction = after
                    break
                continue
            if capture:
                if line.startswith('['):
                    break
                if line.strip():
                    instruction = line.strip()
                    break
        
        return instruction
    
    def _parse_action(
        self,
        step_idx: int,
        action_raw: str,
        state: str,
        avail_actions: List[str],
        action_translate: str,
        action_idx_in_avail: int,
        image: Any,
    ) -> Optional[Action]:
        """
        解析单个 WebShop action
        
        Args:
            step_idx: 步骤索引
            action_raw: 原始动作字符串 (如 "search[xxx]" 或 "click[xxx]")
            state: 当前状态文本
            avail_actions: 可用动作列表
            action_translate: 翻译后的动作 (ASIN -> 商品名)
            action_idx_in_avail: 动作在 avail_actions 中的索引 (-1 表示 search)
            image: 图像特征 (0 或 512 维向量)
        
        Returns:
            Action 或 None
        """
        try:
            # 解析 action_type 和 action_value
            action_type, action_value = self._parse_action_string(action_raw)
            
            # 从 state 中提取 cleaned_html (去掉 Instruction 部分)
            # state 格式: "Amazon Shopping Game\nInstruction:\n...\n[button] Search [button_]\n..."
            cleaned_html = self._extract_cleaned_html(state)
            
            # 处理 candidates 和 target_element
            # - search: 自由输入，没有候选，candidates 为空
            # - click: 用 available_actions，target_element 用 action_translate（商品名版本）
            # 统一用 action_translate 作为 target_element（search 时两者相同）
            if action_type == 'search':
                # search 是自由输入，不是从候选中选择
                candidates = []
            else:
                # click 的 candidates 就是 available_actions（如 ['click[buy now]', ...]）
                candidates = avail_actions if avail_actions else []
            
            # target_element 统一用 action_translate
            target_element = action_translate if action_translate else action_raw
            
            # 图像特征
            screenshot = None
            if isinstance(image, list) and len(image) > 0:
                # 存储为 list (512 维向量)，不是 bytes
                screenshot = image
            
            action = Action(
                action_idx=step_idx,
                action_type=action_type,
                action_value=action_value,
                action_repr=action_raw,  # 原始动作（ASIN 版本）
                cleaned_html=cleaned_html,  # 去掉 instruction 后的内容
                raw_html=state,             # 原始 state 作为 raw_html
                screenshot=screenshot,
                target_element=target_element,  # 商品名版本，能在 candidates 中找到
                candidates=candidates,
                metadata={
                    'action_translate': action_translate,  # 翻译后的动作（商品名版本）
                    'action_idx_in_available': action_idx_in_avail,
                }
            )
            
            return action
            
        except Exception as e:
            print(f"⚠️ Failed to parse action {step_idx}: {e}")
            return None
    
    def _extract_cleaned_html(self, state: str) -> str:
        """
        从 state 中提取 cleaned_html (去掉 Instruction 部分)
        
        state 格式:
            Amazon Shopping Game
            Instruction: 
            i need a high speed usb flash drive...
            [button] Search [button_]
            ...
        
        返回第一个 [ 开始的内容
        """
        # 找到第一个 [ 的位置
        idx = state.find('[')
        if idx != -1:
            return state[idx:]
        else:
            # 如果没找到 [，返回空
            return ''
    
    def _parse_action_string(self, action: str) -> tuple:
        """
        解析动作字符串
        
        Args:
            action: 如 "search[living room furniture]" 或 "click[buy now]"
        
        Returns:
            (action_type, action_value)
            - search: action_value = 搜索内容
            - click:  action_value = "" (空)
        """
        action = action.strip()
        
        if action.startswith('search[') and action.endswith(']'):
            search_query = action[7:-1]
            return ('search', search_query)
        elif action.startswith('click[') and action.endswith(']'):
            # click 没有 value
            return ('click', '')
        else:
            # 其他情况
            return ('unknown', action)


# =============================================================================
# WebLINX Loader
# =============================================================================

class WebLINXLoader(BaseLoader):
    """
    WebLINX 数据集加载器
    
    数据特点：
    - 数据按 action 分割，需要按 demo 聚合成 Record
    - 每条记录包含：demo, turn, action, action_history, utterances, candidates, clean_html, viewport
    - action 类型：click, say, text_input, scroll, load, submit, change
    - train.json.gz 格式（gzip 压缩的 JSONL）
    
    数据路径：
    - /mnt/petrelfs/liuhaoze/datasets/Agent_Data/weblinx/chat_data/data/chat/train.json.gz
    """
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        初始化 WebLINX Loader
        
        Args:
            data_dir: 数据目录，如 /mnt/.../weblinx/chat_data/data/chat
            split: 数据集 split，如 'train', 'valid', 'test' 等
        """
        # 构建文件路径作为 data_path
        filepath = os.path.join(data_dir, f'{split}.json.gz')
        super().__init__(filepath)
        self.data_dir = data_dir
        self.split = split
        self.data = []  # 原始 action 列表
        self.demos = {}  # demo_id -> actions 映射
    
    def load(self) -> List[Dict]:
        """
        加载原始数据并按 demo 聚合
        """
        import gzip
        from collections import defaultdict
        
        # 构建文件路径
        filename = f'{self.split}.json.gz'
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"WebLINX 数据文件不存在: {filepath}")
        
        print(f"📂 Loading WebLINX ({self.split}): {filepath}")
        
        # 加载 gzip 压缩的 JSONL
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        
        # 按 demo 聚合
        self.demos = defaultdict(list)
        for record in self.data:
            self.demos[record['demo']].append(record)
        
        # 按 turn 排序每个 demo 的 actions
        for demo_id in self.demos:
            self.demos[demo_id].sort(key=lambda x: x['turn'])
        
        print(f"✅ Loaded {len(self.data):,} actions from {len(self.demos):,} demos")
        
        return self.data
    
    def parse_record(self, demo_id: str, idx: int = 0) -> Optional[Record]:
        """
        解析单个 demo 为 Record
        
        Args:
            demo_id: demo ID
            idx: Record 索引
        
        Returns:
            Record 或 None
        """
        if demo_id not in self.demos:
            print(f"⚠️ Demo '{demo_id}' not found")
            return None
        
        raw_actions = self.demos[demo_id]
        
        try:
            # 提取完整的 utterances（WebLINX 没有统一的 instruction）
            full_utterances = self._extract_full_utterances(raw_actions)
            
            # 提取网站域名和完整 URL
            website, website_url = self._extract_website(raw_actions)
            
            # 解析每个 action
            actions = []
            for action_idx, raw_action in enumerate(raw_actions):
                action = self._parse_action(raw_action, action_idx)
                if action:
                    actions.append(action)
            
            if not actions:
                return None
            
            record = Record(
                actions=actions,
                sample_id=f"weblinx_{idx}",  # 使用序号，demo_id 放到 metadata
                instruction=None,  # WebLINX 没有统一的 instruction
                website=website,   # 从 load(url=...) 提取的域名，可能为 None
                metadata={
                    'demo_id': demo_id,  # 原生 demo ID
                    'full_utterances': full_utterances,  # 完整的对话历史
                    'website_url': website_url,  # 完整的网站 URL
                }
            )
            
            return record
            
        except Exception as e:
            print(f"⚠️ Failed to parse demo {demo_id}: {e}")
            return None
    
    def _extract_full_utterances(self, raw_actions: List[Dict]) -> str:
        """
        提取完整的 utterances（累积的对话历史）
        
        WebLINX 没有统一的 instruction，而是通过 utterances 逐步交互
        格式: "[-00:08] Hello [00:33] Open momondo.in and login..."
        
        返回最后一个有效的 utterances（最完整的版本）
        """
        full_utterances = ""
        
        # 遍历所有 action，找最后一个有效的 utterances
        for raw_action in raw_actions:
            utterances = raw_action.get('utterances')
            if not utterances or utterances == 'null':
                continue
            
            # 跳过 "N o   i n s t r u c t o r   u t t e r a n c e" 这种无效值
            if 'N o   i n s t r u c t o r   u t t e r a n c e ' in utterances.lower():
                continue
            
            full_utterances = utterances
        
        return full_utterances
    
    def _extract_website(self, raw_actions: List[Dict]) -> tuple:
        """
        从 action_history 中提取主网站域名和完整 URL
        
        策略：
        1. 遍历所有 action 的 action_history
        2. 用正则匹配第一个 load(url="xxx")
        3. 提取 URL 的域名作为主网站
        
        注意：action_history 是滑动窗口，早期的 load 可能被滚出去
        所以需要遍历所有 action 找第一个出现的 URL
        
        Returns:
            (domain, full_url): 域名和完整 URL，找不到则返回 (None, None)
        """
        import re
        from urllib.parse import urlparse
        
        first_url = None
        
        for raw_action in raw_actions:
            action_history = raw_action.get('action_history', '')
            if not action_history or action_history == 'null':
                continue
            
            # 匹配 load(url="xxx") 中的 URL
            match = re.search(r'load\(url="([^"]+)"\)', action_history)
            if match:
                first_url = match.group(1)
                break  # 找到第一个就停止
        
        if not first_url:
            return (None, None)
        
        # 提取域名
        try:
            parsed = urlparse(first_url)
            domain = parsed.netloc
            # 移除 www. 前缀（可选）
            if domain.startswith('www.'):
                domain = domain[4:]
            return (domain, first_url)
        except Exception:
            return (None, first_url)
    
    def _parse_action(self, raw_action: Dict, action_idx: int) -> Optional[Action]:
        """
        解析单个 WebLINX action
        """
        try:
            action_str = raw_action.get('action', '')
            turn = raw_action.get('turn', 0)
            
            # 解析 action 类型和 value
            action_type, action_value = self._parse_action_string(action_str)
            
            # 解析 candidates
            candidates = self._parse_candidates(raw_action.get('candidates', ''))
            
            # target_element: 有 uid 的类型从 action_repr 提取 uid，say 没有 target
            # - click, text_input, change, submit: 有 uid
            # - say, load, scroll: 无 target
            target_element = None
            if action_type in ('click', 'text_input', 'change', 'submit'):
                # 从 action_str 提取 uid
                import re
                uid_match = re.search(r'uid="([^"]*)"', action_str)
                if uid_match and uid_match.group(1) != 'None':
                    target_element = uid_match.group(1)
            
            # clean_html
            clean_html = raw_action.get('clean_html', '')
            
            # viewport
            viewport = raw_action.get('viewport', '')
            
            action = Action(
                action_idx=action_idx,
                action_type=action_type,
                action_value=action_value,
                action_repr=action_str,
                cleaned_html=clean_html,
                raw_html=None,  # WebLINX 没有 raw_html
                screenshot=None,  # WebLINX 没有 screenshot
                target_element=target_element,
                candidates=candidates,
                metadata={
                    'turn': turn,
                    'action_history': raw_action.get('action_history', ''),
                    'utterances': raw_action.get('utterances', ''),
                    'viewport': viewport,
                }
            )
            
            return action
            
        except Exception as e:
            print(f"⚠️ Failed to parse action {action_idx}: {e}")
            return None
    
    def _parse_action_string(self, action_str: str) -> tuple:
        """
        解析 WebLINX action 字符串
        
        Args:
            action_str: 如 'click(uid="xxx")', 'say(speaker="navigator", utterance="...")'
        
        Returns:
            (action_type, action_value)
            - click, submit: value = ""
            - say: value = utterance
            - text_input: value = text
            - load: value = url
            - scroll: value = "x=..., y=..."
            - change: value = value
        """
        import re
        
        action_str = action_str.strip()
        
        # 提取 action type（根据 ( 切分）
        if '(' not in action_str:
            return ('unknown', action_str)
        
        action_type = action_str.split('(')[0]
        action_value = ''
        
        # 根据 action type 提取 value
        if action_type == 'say':
            match = re.search(r'utterance="([^"]*)"', action_str)
            if match:
                action_value = match.group(1)
        elif action_type == 'text_input':
            match = re.search(r'text="([^"]*)"', action_str)
            if match:
                action_value = match.group(1)
        elif action_type == 'load':
            match = re.search(r'url="([^"]*)"', action_str)
            if match:
                action_value = match.group(1)
        elif action_type == 'scroll':
            x_match = re.search(r'x=(-?\d+)', action_str)
            y_match = re.search(r'y=(-?\d+)', action_str)
            x = x_match.group(1) if x_match else '0'
            y = y_match.group(1) if y_match else '0'
            action_value = f"x={x}, y={y}"
        elif action_type == 'change':
            match = re.search(r'value="([^"]*)"', action_str)
            if match:
                action_value = match.group(1)
        # click, submit 没有 value
        
        return (action_type, action_value)
    
    def _parse_candidates(self, candidates_str: str) -> List[Dict]:
        """
        解析 candidates 字符串为元素列表
        
        格式: "(uid = xxx) [[tag]] div [[xpath]] /html/... [[text]] ... [[bbox]] x=... [[attributes]] ... [[children]] ..."
        
        字段统计（基于 181,458 个 candidates）：
        - [[tag]]: 100%
        - [[bbox]]: 100%
        - [[attributes]]: 99.9%
        - [[xpath]]: 99.8%
        - [[text]]: 47.6%
        - [[children]]: 47.5% (子元素标签名列表，如 "div span")
        """
        if not candidates_str or candidates_str == 'null':
            return []
        
        import re
        
        candidates = []
        
        # 按 (uid = 分割
        parts = re.split(r'(?=\(uid = )', candidates_str)
        
        for part in parts:
            part = part.strip()
            if not part.startswith('(uid = '):
                continue
            
            try:
                # 提取 uid
                uid_match = re.match(r'\(uid = ([^)]+)\)', part)
                if not uid_match:
                    continue
                uid = uid_match.group(1).strip()
                
                # 提取其他字段
                cand = {'uid': uid}
                
                # [[tag]]
                tag_match = re.search(r'\[\[tag\]\]\s*(\w+)', part)
                if tag_match:
                    cand['tag'] = tag_match.group(1)
                
                # [[xpath]] - 匹配到下一个 [[field]] 或字符串末尾
                xpath_match = re.search(r'\[\[xpath\]\]\s*(.+?)(?=\s*\[\[|$)', part)
                if xpath_match:
                    cand['xpath'] = xpath_match.group(1).strip()
                
                # [[text]]
                text_match = re.search(r'\[\[text\]\]\s*([^\[]*)', part)
                if text_match:
                    cand['text'] = text_match.group(1).strip()
                
                # [[bbox]]
                bbox_match = re.search(r'\[\[bbox\]\]\s*([^\[]+)', part)
                if bbox_match:
                    cand['bbox'] = bbox_match.group(1).strip()
                
                # [[attributes]] - 解析为字典
                attr_match = re.search(r'\[\[attributes\]\]\s*([^\[]+)', part)
                if attr_match:
                    attr_str = attr_match.group(1).strip()
                    cand['attributes_raw'] = attr_str  # 保留原始字符串
                    cand['attributes'] = self._parse_attributes_string(attr_str)  # 解析为字典
                
                # [[children]] - 子元素标签名列表（如 "div span"）
                children_match = re.search(r'\[\[children\]\]\s*([^\[]*)', part)
                if children_match:
                    children_str = children_match.group(1).strip()
                    if children_str:
                        cand['children'] = children_str
                
                candidates.append(cand)
                
            except Exception:
                continue
        
        return candidates
    
    def _parse_attributes_string(self, attr_str: str) -> Dict[str, str]:
        """
        解析 attributes 字符串为字典
        
        输入格式: "id='xxx' class='yyy zzz' data-webtasks-id='abc...'"
        输出格式: {'id': 'xxx', 'class': 'yyy zzz', 'data-webtasks-id': 'abc...'}
        
        注意：
        - 值可能被截断（包含 ...）
        - 值内可能有空格（如 class）
        - 保留截断的值，用于后续匹配评估
        """
        if not attr_str:
            return {}
        
        import re
        
        result = {}
        
        # 匹配 name='value' 或 name="value"
        # 注意：value 内可能有空格，所以用非贪婪匹配到下一个引号
        pattern = r"([a-zA-Z0-9_-]+)='([^']*)'"
        matches = re.findall(pattern, attr_str)
        
        for name, value in matches:
            result[name] = value
        
        # 也匹配双引号的情况
        pattern_double = r'([a-zA-Z0-9_-]+)="([^"]*)"'
        matches_double = re.findall(pattern_double, attr_str)
        
        for name, value in matches_double:
            if name not in result:  # 避免重复
                result[name] = value
        
        return result
    
    def parse_all(self, show_progress: bool = True) -> List[Record]:
        """
        解析所有 demo 为 Record 列表
        """
        if not self.demos:
            self.load()
        
        records = []
        demo_ids = list(self.demos.keys())
        
        if show_progress:
            from tqdm import tqdm
            demo_ids = tqdm(demo_ids, desc="Parsing demos")
        
        for idx, demo_id in enumerate(demo_ids):
            record = self.parse_record(demo_id, idx)
            if record:
                records.append(record)
        
        return records


# =============================================================================
# 便捷函数
# =============================================================================

def load_mind2web(path: str, show_progress: bool = True) -> List[Record]:
    """便捷函数：加载 Mind2Web 数据集"""
    loader = Mind2WebLoader(path)
    return loader.parse_all(show_progress)


def load_multimodal_mind2web(data_dir: str, split: str = 'train', show_progress: bool = True) -> List[Record]:
    """便捷函数：加载 Multimodal Mind2Web 数据集"""
    loader = MultimodalMind2WebLoader(data_dir, split)
    return loader.parse_all(show_progress)


def load_webshop(path: str, show_progress: bool = True) -> List[Record]:
    """便捷函数：加载 WebShop 数据集"""
    loader = WebShopLoader(path)
    return loader.parse_all(show_progress)


def load_weblinx(data_dir: str, split: str = 'train', show_progress: bool = True) -> List[Record]:
    """便捷函数：加载 WebLINX 数据集"""
    loader = WebLINXLoader(data_dir, split)
    return loader.parse_all(show_progress)


# =============================================================================
# 测试
# =============================================================================

def print_record(record: Record, max_html_len: int = 500):
    """打印 Record 详情"""
    print("=" * 80)
    print(f"📋 Record")
    print("=" * 80)
    
    # 基本信息
    print(f"\n📌 sample_id: {record.sample_id}")
    print(f"📝 Instruction: {record.instruction}")
    print(f"🌐 Website: {record.website}")
    print(f"📊 Metadata: {record.metadata}")
    
    # Actions
    print(f"\n🎬 Actions ({len(record.actions)}):")
    for i, action in enumerate(record.actions):
        print(f"\n  [{i}] Action:")
        print(f"      action_idx: {action.action_idx}")
        print(f"      action_type: {action.action_type}")
        print(f"      action_repr: {action.action_repr}")
        
        # cleaned_html (截断)
        html_preview = action.cleaned_html[:max_html_len] + "..." if len(action.cleaned_html) > max_html_len else action.cleaned_html
        print(f"      cleaned_html: {html_preview[:100]}...")
        
        # target_element
        if action.target_element:
            print(f"      target_element: {action.target_element}")
        
        # candidates
        print(f"      candidates: {len(action.candidates)} items")
        
        # metadata
        if action.metadata:
            print(f"      metadata: {action.metadata}")
    
    print("\n" + "=" * 80)


def print_action(action: Action, max_html_len: int = 2000):
    """打印单个 Action 详情"""
    print("=" * 80)
    print(f"🎬 Action [{action.action_idx}]")
    print("=" * 80)
    
    print(f"\n📌 action_type: {action.action_type}")
    print(f"📌 action_repr: {action.action_repr}")
    
    # cleaned_html
    print(f"\n📄 cleaned_html ({len(action.cleaned_html)} chars):")
    html_preview = action.cleaned_html[:max_html_len]
    if len(action.cleaned_html) > max_html_len:
        html_preview += f"\n... ({len(action.cleaned_html) - max_html_len} more chars)"
    print(html_preview)
    
    # raw_html
    if action.raw_html:
        print(f"\n📄 raw_html: {len(action.raw_html)} chars")
    
    # target_element
    print(f"\n🎯 target_element:")
    if action.target_element:
        for k, v in action.target_element.items():
            print(f"    {k}: {v}")
    else:
        print("    None")
    
    # candidates
    print(f"\n📋 candidates ({len(action.candidates)} items):")
    for i, cand in enumerate(action.candidates[:5]):  # 只显示前 5 个
        print(f"  [{i}] tag={cand.get('tag', 'N/A')}, backend_node_id={cand.get('backend_node_id', 'N/A')}")
    if len(action.candidates) > 5:
        print(f"  ... and {len(action.candidates) - 5} more")
    
    # metadata
    print(f"\n📊 metadata:")
    for k, v in action.metadata.items():
        print(f"    {k}: {v}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    # 测试 Mind2Web Loader
    print("\n" + "=" * 80)
    print("Testing Mind2Web Loader")
    print("=" * 80)
    
    mind2web_path = '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/Mind2Web/data/train_all.json'
    
    if os.path.exists(mind2web_path):
        loader = Mind2WebLoader(mind2web_path)
        loader.load()
        
        # 解析第 1 条，完整展示
        if loader.data:
            record = loader.parse_record(loader.data[0], 0)
            if record:
                print_record(record)
                
                # 打印第一个 action 详情
                if record.actions:
                    print("\n\n")
                    print_action(record.actions[0])
    else:
        print(f"❌ File not found: {mind2web_path}")
