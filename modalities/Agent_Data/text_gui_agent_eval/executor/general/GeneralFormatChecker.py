#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用格式检查器

纯粹基于 data_types.py 合同 (Record / Action) 进行格式检查，
不依赖任何数据集特有逻辑。

适用场景：
- 用户按照 data_types.py 合同自行生成的数据集
- 任何已通过 Loader 转换为 Record 的数据集的基线检查
- 作为自定义 DatasetSpecificChecker 的参考实现
"""

import os
import sys
from typing import List, Tuple, Set

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from text_gui_executor import FormatChecker
from data_types import Record, Action


class GeneralFormatChecker(FormatChecker):
    """
    通用格式检查器：基于 data_types.py 合同本身进行检查

    检查项（全部来源于 dataclass 字段定义）：

    1. Record 层
       - actions (List[Action], 必需): 非空列表
       - instruction (Optional[str]): 如存在则非空 (warning)
       - sample_id (Optional[str]): 建议存在 (warning)

    2. Action 层
       - action_idx (int, 必需): 非负整数
       - action_type (str, 必需): 非空
       - action_repr (str): 建议非空 (warning)
       - cleaned_html (str): 建议非空 (warning, say/scroll 等可能为空)

    3. 全局一致性
       - action_idx 连续性 (warning)
       - action_idx 无重复 (error)
    """

    def check(self, record: Record) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []

        # =================================================================
        # 1. Record 层
        # =================================================================
        if not record.actions:
            errors.append("Record has no actions")
            return errors, warnings

        if record.instruction is not None and not record.instruction.strip():
            warnings.append("instruction is present but empty string")

        if not record.sample_id:
            warnings.append("sample_id is missing")

        # =================================================================
        # 2. Action 层
        # =================================================================
        seen_indices: Set[int] = set()

        for i, action in enumerate(record.actions):
            prefix = f"Action[{i}]"

            # action_type (str, 必需)
            if not action.action_type:
                errors.append(f"{prefix}: action_type is empty")

            # action_idx 非负
            if action.action_idx < 0:
                errors.append(f"{prefix}: action_idx is negative ({action.action_idx})")

            # action_idx 无重复
            if action.action_idx in seen_indices:
                errors.append(f"{prefix}: duplicate action_idx ({action.action_idx})")
            seen_indices.add(action.action_idx)

            # action_repr 建议非空
            if not action.action_repr:
                warnings.append(f"{prefix}: action_repr is empty")

            # cleaned_html 建议非空（say/scroll 等可能合理地为空）
            if not action.cleaned_html:
                warnings.append(f"{prefix}: cleaned_html is empty")

        # =================================================================
        # 3. 全局一致性: action_idx 连续性
        # =================================================================
        if len(record.actions) > 1:
            expected = list(range(len(record.actions)))
            actual = [a.action_idx for a in record.actions]
            if actual != expected:
                warnings.append(
                    f"action_idx not sequential 0..{len(record.actions)-1}: "
                    f"got {actual}"
                )

        return errors, warnings
