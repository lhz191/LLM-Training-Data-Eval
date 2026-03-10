#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX 多轮格式检查器

对整个 Session 进行格式检查，包括：
1. Session 级别：rounds 非空、session_id 存在
2. 每轮 Record 级别：复用单轮检查逻辑
3. 跨轮一致性：website 一致性、instruction 连贯性
"""

import re
import os
import sys
from typing import Dict, List, Tuple, Any

_current_file = os.path.abspath(__file__)
_weblinx_dir = os.path.dirname(_current_file)
_executor_dir = os.path.dirname(_weblinx_dir)
_mt_eval_dir = os.path.dirname(_executor_dir)

if _mt_eval_dir not in sys.path:
    sys.path.insert(0, _mt_eval_dir)

from data_types import Session
from mt_executor import SessionFormatChecker

from .constants import (
    UID_REQUIRED_ACTIONS,
    VALUE_REQUIRED_ACTIONS,
    VALID_ACTION_TYPES,
)


class WebLINXSessionFormatChecker(SessionFormatChecker):
    """
    WebLINX 多轮数据格式检查器

    检查项：
    1. Session 级别
       - rounds 非空
       - session_id 存在
       - metadata 中有 demo_id

    2. 每轮 Record 级别（与单轮 WebLINXFormatChecker 逻辑一致）
       - actions 非空
       - action_type 合法
       - 需要 uid 的操作有 target_element
       - 需要 value 的操作有 action_value
       - candidates 与 uid 一致性

    3. 跨轮一致性
       - website 是否一致
       - 是否存在空轮（无 action 的 Round）
    """

    def check_session(self, session: Session) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """检查整个 Session 的数据格式"""
        errors = []
        warnings = []

        # === 1. Session 级别检查 ===
        if not session.rounds:
            errors.append("Session has no rounds")
            return errors, warnings, {'total_rounds': 0}

        if not session.session_id:
            warnings.append("session_id is missing")

        demo_id = session.metadata.get('demo_id', '')
        if not demo_id:
            warnings.append("Session metadata missing 'demo_id'")

        # === 2. 逐轮检查 ===
        round_stats = []
        total_actions = 0
        total_errors = 0
        websites = set()

        for r_idx, record in enumerate(session.rounds):
            r_errors, r_warnings = self._check_record(record, r_idx)
            errors.extend(r_errors)
            warnings.extend(r_warnings)

            n_actions = len(record.actions) if record.actions else 0
            total_actions += n_actions
            total_errors += len(r_errors)

            if record.website:
                websites.add(record.website)

            round_stats.append({
                'round_idx': r_idx,
                'instruction': record.instruction,
                'num_actions': n_actions,
                'num_errors': len(r_errors),
            })

        # === 3. 跨轮一致性 ===
        if len(websites) > 1:
            warnings.append(f"Inconsistent websites across rounds: {websites}")

        empty_rounds = [s['round_idx'] for s in round_stats if s['num_actions'] == 0]
        if empty_rounds:
            errors.extend([f"Round[{i}]: has no actions" for i in empty_rounds])

        no_instruction = [s['round_idx'] for s in round_stats if not s['instruction']]
        if no_instruction:
            warnings.append(f"Rounds without instruction: {no_instruction}")

        stats = {
            'total_rounds': len(session.rounds),
            'total_actions': total_actions,
            'total_errors': len(errors),
            'total_warnings': len(warnings),
            'round_stats': round_stats,
        }

        return errors, warnings, stats

    def _check_record(self, record, r_idx: int) -> Tuple[List[str], List[str]]:
        """检查单轮 Record 的格式（与单轮 WebLINXFormatChecker 逻辑一致）"""
        errors = []
        warnings = []
        prefix = f"Round[{r_idx}]"

        if not record.actions:
            return errors, warnings

        for i, action in enumerate(record.actions):
            a_errors, a_warnings = self._check_action(action, i, prefix)
            errors.extend(a_errors)
            warnings.extend(a_warnings)

        return errors, warnings

    def _check_action(self, action, idx: int, round_prefix: str) -> Tuple[List[str], List[str]]:
        """检查单个 Action 的格式"""
        errors = []
        warnings = []
        prefix = f"{round_prefix}.Action[{idx}]"

        action_type = action.action_type
        action_value = action.action_value
        action_repr = action.action_repr
        target_element = action.target_element
        candidates = action.candidates

        if not action_type:
            errors.append(f"{prefix}: missing 'action_type'")
        elif action_type == 'unknown':
            errors.append(f"{prefix}: unknown action type, action_repr='{action_repr or 'N/A'}'")
        elif action_type not in VALID_ACTION_TYPES:
            errors.append(f"{prefix}: invalid action_type '{action_type}'")

        if action_type in UID_REQUIRED_ACTIONS:
            if not target_element:
                errors.append(f"{prefix}: {action_type} missing or invalid uid")

        if action_type in VALUE_REQUIRED_ACTIONS:
            if not action_value:
                errors.append(f"{prefix}: {action_type} missing value")

        if target_element and candidates:
            uid_found = self._check_uid_in_candidates(target_element, candidates)
            if not uid_found:
                errors.append(f"{prefix}: target uid not found in candidates")

        if not action.cleaned_html:
            warnings.append(f"{prefix}: empty 'clean_html'")

        return errors, warnings

    def _check_uid_in_candidates(self, target_uid: str, candidates) -> bool:
        if not target_uid or not candidates:
            return False
        if isinstance(candidates, str):
            return target_uid in candidates
        if isinstance(candidates, list):
            for cand in candidates:
                if isinstance(cand, dict) and cand.get('uid') == target_uid:
                    return True
        return False
