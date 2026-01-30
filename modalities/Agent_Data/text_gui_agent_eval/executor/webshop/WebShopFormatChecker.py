#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebShop 格式检查器
"""

import os
import sys
import re
from typing import List, Tuple

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from text_gui_executor import FormatChecker
from data_types import Record, Action

class WebShopFormatChecker(FormatChecker):
    """
    WebShop 数据格式检查器
    
    检查项：
    1. Record 级别
       - instruction 是否存在且非空
       - actions 是否存在且非空
       
    2. Action 级别
       - action 格式是否正确（search[xxx] 或 click[xxx]）
       - action_type 是否为 search 或 click
       - cleaned_html (state) 是否存在
       - 对于 click 操作，target 是否在 candidates (available_actions) 中
    """
    
    def check(self, record: Record) -> Tuple[List[str], List[str]]:
        """检查 WebShop Record 的数据格式"""
        errors = []
        warnings = []
        
        # === 1. Record 级别检查 ===
        
        # instruction
        if not record.instruction or not record.instruction.strip():
            errors.append("Record has empty 'instruction'")
        
        # actions
        if not record.actions:
            errors.append("Record has no actions")
            return errors, warnings
        
        # === 2. Action 级别检查 ===
        for i, action in enumerate(record.actions):
            action_errors, action_warnings = self._check_action(action, i)
            errors.extend(action_errors)
            warnings.extend(action_warnings)
        
        return errors, warnings
    
    def _check_action(self, action: Action, idx: int) -> Tuple[List[str], List[str]]:
        """检查单个 Action 的格式"""
        errors = []
        warnings = []
        prefix = f"Action[{idx}]"
        
        # action_type
        action_type = action.action_type
        if action_type not in ('search', 'click'):
            errors.append(f"{prefix}: invalid action_type '{action_type}', expected 'search' or 'click'")
        
        # action_repr 格式检查
        action_repr = action.action_repr
        if action_repr:
            if action_type == 'search':
                if not action_repr.startswith('search[') or not action_repr.endswith(']'):
                    errors.append(f"{prefix}: invalid search format: {action_repr}")
            elif action_type == 'click':
                if not action_repr.startswith('click[') or not action_repr.endswith(']'):
                    errors.append(f"{prefix}: invalid click format: {action_repr}")
        else:
            errors.append(f"{prefix}: missing 'action_repr'")
        
        # cleaned_html (state)
        state = action.cleaned_html or ''
        if not state:
            errors.append(f"{prefix}: empty 'cleaned_html' (state)")
        
        # 对于 search 操作，检查是否有搜索内容
        if action_type == 'search':
            if not action.action_value or not action.action_value.strip():
                errors.append(f"{prefix}: search must have action_value (search content)")
            
            # 检查 [button] Search [button_] 是否在 state 中
            if state and '[button] search [button_]' not in state.lower():
                errors.append(f"{prefix}: search button not found in state")
        
        # 对于 click 操作，检查 target 是否在 candidates 中
        elif action_type == 'click':
            target = action.target_element
            candidates = action.candidates
            
            if target and candidates:
                # WebShop 的 candidates 是 available_actions 列表
                # target_element 是 action_translate（商品名版本）
                if target not in candidates:
                    # 尝试匹配原始 action_repr
                    if action_repr not in candidates:
                        errors.append(f"{prefix}: target not in available_actions")
            
            # 检查 action_repr 是否在 state 中可定位
            # 格式: [button] xxx [button_] 或 [clicked button] xxx [clicked button_]
            if state and action_repr:
                click_target = action_repr[6:-1].lower() if action_repr.startswith('click[') else ''
                if click_target:
                    state_lower = state.lower()
                    pattern1 = f'[button] {click_target} [button_]'
                    pattern2 = f'[clicked button] {click_target} [clicked button_]'
                    if pattern1 not in state_lower and pattern2 not in state_lower:
                        errors.append(f"{prefix}: target not found in state")
        
        return errors, warnings


