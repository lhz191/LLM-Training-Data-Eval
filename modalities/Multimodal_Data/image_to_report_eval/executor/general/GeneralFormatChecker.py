#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用格式检查器

纯粹基于 data_types.py 合同 (ImageToReportSample) 进行格式检查，
不依赖任何数据集特有逻辑。

适用场景：
- 任何已通过 Loader 转换为 ImageToReportSample 的数据集的基线检查
- 用户自行生成的 image-to-report 数据集
- 作为数据集特有 FormatChecker 的参考实现

检查项（全部来源于 dataclass 字段定义）：

1. ImageToReportSample 层
   - sample_id (str): 非空
   - instruction (str, 必需): 非空
   - report (str, 必需): 非空
   - images (List[str], 必需): 非空列表，每个路径非空

2. 一致性检查
   - <image> token 数量与 images 列表长度是否匹配
"""

import os
import sys
from typing import List, Tuple

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import ImageToReportSample
from report_executor import FormatChecker


class GeneralFormatChecker(FormatChecker):
    """
    通用格式检查器：基于 data_types.py 合同本身进行检查

    检查项：
    1. sample_id 非空 (error)
    2. instruction 非空 (error)
    3. report 非空 (error)
    4. images 非空列表，每个路径非空 (error)
    5. <image> token 数量 vs images 列表长度 (error)
    """

    def check(self, sample: ImageToReportSample) -> Tuple[List[str], List[str]]:
        errors = []
        warnings = []

        # =================================================================
        # 1. 必需字段非空
        # =================================================================

        if not sample.sample_id:
            errors.append("sample_id 为空")

        if not sample.instruction or not sample.instruction.strip():
            errors.append("instruction 为空")

        if not sample.report or not sample.report.strip():
            errors.append("report 为空")

        if not sample.images:
            errors.append("images 列表为空")
        else:
            for i, path in enumerate(sample.images):
                if not path or not path.strip():
                    errors.append(f"images[{i}] 路径为空")

        # =================================================================
        # 2. 一致性
        # =================================================================

        token_count = sample.instruction.count("<image>") if sample.instruction else 0
        image_count = len(sample.images)
        if token_count != image_count:
            errors.append(
                f"<image> token 数量({token_count}) != 图片数量({image_count})"
            )

        return errors, warnings
