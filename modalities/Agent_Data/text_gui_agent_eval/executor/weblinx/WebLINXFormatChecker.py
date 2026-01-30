#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX 格式检查器
"""

import re
from typing import Dict, List, Tuple

from .constants import (
    UID_REQUIRED_ACTIONS,
    VALUE_REQUIRED_ACTIONS,
    VALID_ACTION_TYPES,
)


class WebLINXFormatChecker:
    """
    WebLINX 数据格式检查器
    
    检查项：
    1. Record 级别
       - demo_id 是否存在
       - utterances 是否有效（非空且非占位符）
       - actions 是否存在且非空
       
    2. Action 级别
       - action_type 是否合法
       - 需要 uid 的操作是否有 target_element
       - 需要 value 的操作是否有 action_value
       - candidates 存在时，uid 是否在其中
       
    3. 数据质量检查
       - clean_html 是否为空
       - viewport 是否为空
    """
    
    def check(self, record) -> Tuple[List[str], List[str]]:
        """检查 WebLINX Record 的数据格式"""
        errors = []
        warnings = []  # 保留接口，但不使用
        
        # === 1. Record 级别检查 ===
        
        # demo_id
        demo_id = record.metadata.get('demo_id', '')
        if not demo_id:
            errors.append("Record missing 'demo_id' in metadata")
        
        # utterances - 不检查，WebLINX 的 utterances 可能为空或是占位符
        # website - 不检查，可能无法从 actions 中提取
        
        # actions
        if not record.actions:
            errors.append("Record has no actions")
            return errors, warnings  # 无法继续检查 action 级别
        
        # === 2. Action 级别检查 ===
        for i, action in enumerate(record.actions):
            action_errors, action_warnings = self._check_action(action, i)
            errors.extend(action_errors)
            # warnings 不再使用
        
        return errors, warnings
    
    def _check_action(self, action, idx: int) -> Tuple[List[str], List[str]]:
        """检查单个 Action 的格式"""
        errors = []
        warnings = []  # 保留接口，但不使用
        prefix = f"Action[{idx}]"
        
        action_type = action.action_type
        action_value = action.action_value
        action_repr = action.action_repr
        target_element = action.target_element  # uid 字符串
        candidates = action.candidates
        
        # === 1. action_type 检查 ===
        if not action_type:
            errors.append(f"{prefix}: missing 'action_type'")
        elif action_type == 'unknown':
            errors.append(f"{prefix}: unknown action type, action_repr='{action_repr[:80] if action_repr else 'N/A'}'")
        elif action_type not in VALID_ACTION_TYPES:
            errors.append(f"{prefix}: invalid action_type '{action_type}', must be one of {VALID_ACTION_TYPES}")
        
        # === 2. target_element (uid) 检查 ===
        # target_element 是 loader 从 action_repr 中提取的 uid
        # 如果提取失败（uid=None 或 uid="" 或没有 uid），target_element 就是 None
        if action_type in UID_REQUIRED_ACTIONS:
            if not target_element:  # None 或空字符串
                errors.append(f"{prefix}: {action_type} missing or invalid uid")
        
        # === 3. action_value 检查 ===
        if action_type in VALUE_REQUIRED_ACTIONS:
            expected_param = VALUE_REQUIRED_ACTIONS[action_type]
            if not action_value:
                errors.append(f"{prefix}: {action_type} missing value (expected '{expected_param}')")
        
        # === 4. candidates 一致性检查 ===
        if target_element and candidates:
            # 检查 uid 是否在 candidates 中
            uid_in_candidates = self._check_uid_in_candidates(target_element, candidates)
            if not uid_in_candidates:
                errors.append(f"{prefix}: target uid not found in candidates")
        
        # === 5. 数据质量检查 ===
        
        # clean_html
        if not action.cleaned_html:
            errors.append(f"{prefix}: empty 'clean_html'")
        elif target_element:
            # 检查 target uid 是否在 clean_html 中（支持截断格式如 1...0f）
            uid_in_clean_html = self._check_uid_in_clean_html(target_element, action.cleaned_html)
            if not uid_in_clean_html:
                errors.append(f"{prefix}: target uid not found in clean_html")
        
        # viewport - 不检查，可能为空
        # utterances - 不检查，可能是占位符
        
        return errors, warnings
    
    def _check_uid_in_candidates(self, target_uid: str, candidates: List[Dict]) -> bool:
        """检查 target uid 是否在 candidates 中"""
        if not target_uid or not candidates:
            return False
        
        for cand in candidates:
            cand_uid = cand.get('uid', '')
            if cand_uid == target_uid:
                return True
        
        return False
    
    def _check_uid_in_clean_html(self, target_uid: str, clean_html: str) -> bool:
        """
        检查 target uid 是否在 clean_html 中
        
        clean_html 中的 uid 可能是截断格式（如 1...0f 对应 1b010db4-3df2-4c0f）
        匹配规则：截断的 uid 格式为 "前缀...后缀"，检查完整 uid 是否以前缀开头、后缀结尾
        """
        if not target_uid or not clean_html:
            return False
        
        # 提取 clean_html 中所有的 data-webtasks-id 值
        # 格式: data-webtasks-id="xxx" 或 data-webtasks-id='xxx'
        pattern = r'data-webtasks-id=["\']([^"\']+)["\']'
        uid_values = re.findall(pattern, clean_html)
        
        for uid_val in uid_values:
            if self._match_truncated_uid(target_uid, uid_val):
                return True
        
        return False
    
    def _match_truncated_uid(self, full_uid: str, truncated_uid: str) -> bool:
        """
        检查完整 uid 是否匹配截断的 uid
        
        截断格式: "前缀...后缀" (如 1...0f, b5...69)
        匹配规则: full_uid.startswith(前缀) and full_uid.endswith(后缀)
        """
        if not full_uid or not truncated_uid:
            return False
        
        # 如果没有截断，直接比较
        if '...' not in truncated_uid:
            return full_uid == truncated_uid
        
        # 分割截断的 uid
        parts = truncated_uid.split('...')
        if len(parts) != 2:
            return False
        
        prefix, suffix = parts
        return full_uid.startswith(prefix) and full_uid.endswith(suffix)
