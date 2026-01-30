#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mind2Web 格式检查器
"""

import os
import sys
from typing import List, Tuple, Dict, Any

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from text_gui_executor import FormatChecker
from data_types import Record, Action

class Mind2WebFormatChecker(FormatChecker):
    """
    Mind2Web 数据格式检查器
    
    检查项：
    1. Record 级别
       - annotation_id 是否存在
       - instruction 是否存在且非空
       - actions 是否存在且非空
       
    2. Action 级别
       - action_uid 是否存在
       - target_element 是否存在
       - operation 是否存在（op 字段）
       - candidates 是否存在
       
    3. 数据一致性检查
       - target 是否在 candidates 中（通过 backend_node_id 匹配）
       - backend_node_id 是否在 cleaned_html 中可找到
    """
    
    def check(self, record: Record) -> Tuple[List[str], List[str]]:
        """检查 Mind2Web Record 的数据格式"""
        errors = []
        
        # === 1. Record 级别检查 ===
        
        # annotation_id
        annotation_id = record.metadata.get('annotation_id', '')
        if not annotation_id:
            errors.append("Record missing 'annotation_id' in metadata")
        
        # instruction
        if not record.instruction or not record.instruction.strip():
            errors.append("Record has empty 'instruction'")
        
        # actions
        if not record.actions:
            errors.append("Record has no actions")
            return errors, []  # 无法继续检查 action 级别
        
        # === 2. Action 级别检查 ===
        for i, action in enumerate(record.actions):
            action_errors, _ = self._check_action(action, i)
            errors.extend(action_errors)
        
        return errors, []
    
    def _check_action(self, action: Action, idx: int) -> Tuple[List[str], List[str]]:
        """检查单个 Action 的格式"""
        errors = []
        prefix = f"Action[{idx}]"
        
        # action_uid
        action_uid = action.metadata.get('action_uid', '')
        if not action_uid:
            errors.append(f"{prefix}: missing 'action_uid'")
        
        # target_element
        target = action.target_element
        if not target:
            errors.append(f"{prefix}: missing 'target_element'")
        else:
            # 检查 backend_node_id
            backend_node_id = target.get('backend_node_id')
            if not backend_node_id:
                errors.append(f"{prefix}: target_element missing 'backend_node_id'")
        
        # operation
        operation = action.metadata.get('operation', {})
        if not operation:
            errors.append(f"{prefix}: missing 'operation' in metadata")
        else:
            op = operation.get('op', '').upper()
            value = operation.get('value', '')
            
            if not op:
                errors.append(f"{prefix}: operation missing 'op' field")
            else:
                # 根据操作类型检查 value
                # CLICK: 不应该有 value
                # SELECT/TYPE: 必须有 value
                if op == 'CLICK':
                    if value and value.strip():
                        errors.append(f"{prefix}: CLICK should not have value, got '{value[:30]}'")
                elif op in ('SELECT', 'TYPE'):
                    if not value or not value.strip():
                        errors.append(f"{prefix}: {op} must have value")
        
        # candidates
        candidates = action.candidates
        if not candidates:
            errors.append(f"{prefix}: no candidates")
        else:
            # === 3. 数据一致性检查 ===
            # 检查 target 是否在 candidates 中
            if target:
                target_in_candidates = self._check_target_in_candidates(target, candidates)
                if not target_in_candidates:
                    errors.append(f"{prefix}: target not found in candidates")
        
        # cleaned_html
        if not action.cleaned_html:
            errors.append(f"{prefix}: empty 'cleaned_html'")
        else:
            # 检查 backend_node_id 是否在 cleaned_html 中可找到
            if target:
                backend_node_id = target.get('backend_node_id')
                if backend_node_id:
                    # 在 cleaned_html 中搜索 backend_node_id
                    # 可能的格式: backend_node_id="136" 或 data-backend-node-id="136" 或直接作为某个属性值
                    node_id_str = str(backend_node_id)
                    if node_id_str not in action.cleaned_html:
                        errors.append(f"{prefix}: backend_node_id not found in cleaned_html")
        
        return errors, []
    
    def _check_target_in_candidates(self, target: Dict, candidates: List[Dict]) -> bool:
        """检查 target 是否在 candidates 中（通过 backend_node_id 匹配）"""
        target_node_id = target.get('backend_node_id')
        if not target_node_id:
            return False
        
        for cand in candidates:
            if cand.get('backend_node_id') == target_node_id:
                return True
        
        return False


# =============================================================================
# HTML 定位器
# =============================================================================

