#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用格式检查器

纯粹基于 data_types.py 合同 (MathSample) 进行格式检查，
不依赖任何数据集特有逻辑。

适用场景：
- 用户按照 data_types.py 合同自行生成的数据集
- 任何已通过 Loader 转换为 MathSample 的数据集的基线检查
- 作为自定义 DatasetSpecificChecker 的参考实现
"""

import os
import sys
from typing import List, Tuple

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code_executor import FormatChecker
from data_types import MathSample


class GeneralFormatChecker(FormatChecker):
    """
    通用格式检查器：基于 data_types.py 合同本身进行检查

    检查项（全部来源于 MathSample dataclass 字段定义）：

    1. question (str, 必需): 非空
    2. solution (Union[str, List[str]], 必需):
       - str: 非空
       - List[str]: 非空列表，每个元素非空
    3. ground_truth (Union[str, List[Any], None]):
       - 如果为 str: 非空 (warning)
       - 如果为 list: 非空列表 (warning)
       - None: 允许（某些数据集不提供 GT）(warning)
    4. sample_id (Optional[str]): 建议存在 (warning)
    5. source_dataset (Optional[str]): 建议存在 (warning)

    不检查的内容（属于 dataset-specific 范畴）：
    - solution 中的代码格式 (如 <llm-code> 标签、```python 块)
    - solution 中的答案格式 (如 \\boxed{}, ####, The answer is:)
    - ground_truth 与 solution 的一致性
    """

    def check(self, sample: MathSample) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []

        # =================================================================
        # 1. question (str, 必需)
        # =================================================================
        if not sample.question or not sample.question.strip():
            errors.append("question is empty or missing")

        # =================================================================
        # 2. solution (Union[str, List[str]], 必需)
        # =================================================================
        sol = sample.solution
        if sol is None:
            errors.append("solution is None")
        elif isinstance(sol, str):
            if not sol.strip():
                errors.append("solution is empty string")
        elif isinstance(sol, list):
            if len(sol) == 0:
                errors.append("solution is empty list")
            else:
                for i, prog in enumerate(sol):
                    if not isinstance(prog, str):
                        errors.append(f"solution[{i}]: expected str, got {type(prog).__name__}")
                    elif not prog.strip():
                        warnings.append(f"solution[{i}]: empty string in program list")
        else:
            errors.append(f"solution: unexpected type {type(sol).__name__}, expected str or List[str]")

        # =================================================================
        # 3. ground_truth (Union[str, List[Any], None])
        # =================================================================
        gt = sample.ground_truth
        if gt is None:
            warnings.append("ground_truth is None (no GT provided)")
        elif isinstance(gt, str):
            if not gt.strip():
                warnings.append("ground_truth is empty string")
        elif isinstance(gt, list):
            if len(gt) == 0:
                warnings.append("ground_truth is empty list")
        # other types: allow (could be numeric, etc.)

        # =================================================================
        # 4. sample_id
        # =================================================================
        if not sample.sample_id:
            warnings.append("sample_id is missing")

        # =================================================================
        # 5. source_dataset
        # =================================================================
        if not sample.source_dataset:
            warnings.append("source_dataset is missing")

        return errors, warnings
