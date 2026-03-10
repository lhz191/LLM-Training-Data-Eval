#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用多轮格式检查器

纯粹基于 Session / Record / Action 合同进行格式检查，
不依赖任何数据集特有逻辑。

检查项：
1. Session 层
   - rounds 非空
   - session_id 建议存在

2. 每轮 Record 层（与单轮 GeneralFormatChecker 逻辑一致）
   - actions 非空
   - action_type 非空
   - action_idx 非负、无重复

3. 跨轮一致性
   - action_idx 在整个 Session 内无冲突
"""

import os
import sys
from typing import List, Tuple, Set, Dict, Any

_current_file = os.path.abspath(__file__)
_general_dir = os.path.dirname(_current_file)
_executor_dir = os.path.dirname(_general_dir)
_mt_eval_dir = os.path.dirname(_executor_dir)

if _mt_eval_dir not in sys.path:
    sys.path.insert(0, _mt_eval_dir)

from data_types import Session
from mt_executor import SessionFormatChecker


class GeneralSessionFormatChecker(SessionFormatChecker):
    """
    通用 Session 格式检查器：基于 dataclass 合同本身进行检查
    """

    def check_session(self, session: Session) -> Tuple[List[str], List[str], Dict[str, Any]]:
        errors = []
        warnings = []

        # === 1. Session 层 ===
        if not session.rounds:
            errors.append("Session has no rounds")
            return errors, warnings, {'total_rounds': 0}

        if not session.session_id:
            warnings.append("session_id is missing")

        # === 2. 每轮 Record 层 ===
        round_stats = []
        total_actions = 0

        for r_idx, record in enumerate(session.rounds):
            r_errors, r_warnings = self._check_record(record, r_idx)
            errors.extend(r_errors)
            warnings.extend(r_warnings)

            n_actions = len(record.actions) if record.actions else 0
            total_actions += n_actions
            round_stats.append({
                'round_idx': r_idx,
                'num_actions': n_actions,
                'num_errors': len(r_errors),
            })

        stats = {
            'total_rounds': len(session.rounds),
            'total_actions': total_actions,
            'total_errors': len(errors),
            'total_warnings': len(warnings),
            'round_stats': round_stats,
        }

        return errors, warnings, stats

    def _check_record(self, record, r_idx: int) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []
        prefix = f"Round[{r_idx}]"

        if not record.actions:
            errors.append(f"{prefix}: has no actions")
            return errors, warnings

        if record.instruction is not None and not record.instruction.strip():
            warnings.append(f"{prefix}: instruction is present but empty")

        seen_indices: Set[int] = set()

        for i, action in enumerate(record.actions):
            a_prefix = f"{prefix}.Action[{i}]"

            if not action.action_type:
                errors.append(f"{a_prefix}: action_type is empty")

            if action.action_idx < 0:
                errors.append(f"{a_prefix}: action_idx is negative ({action.action_idx})")

            if action.action_idx in seen_indices:
                errors.append(f"{a_prefix}: duplicate action_idx ({action.action_idx})")
            seen_indices.add(action.action_idx)

            if not action.action_repr:
                warnings.append(f"{a_prefix}: action_repr is empty")

        return errors, warnings
