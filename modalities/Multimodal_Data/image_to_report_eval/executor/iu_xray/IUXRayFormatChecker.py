#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IU X-Ray 数据集格式检查器

继承链：
  common.base.BaseFormatChecker
    └── report_executor.FormatChecker  (modality='image_to_report')
          └── IUXRayFormatChecker

IU X-Ray 原始格式 (JSONL):
{
    "query": "<image><image>Please review the chest X-ray image and create a report...",
    "response": "The heart size and pulmonary vascularity appear within normal limits...",
    "images": ["/iu_xray/image/CXR2384_IM-0942/0.png", "/iu_xray/image/CXR2384_IM-0942/1.png"]
}

检查项：

1. ImageToReportSample 层
   - sample_id (str): 非空
   - instruction (str, 必需): 非空
   - report (str, 必需): 非空
   - images (List[str], 必需): 非空列表，每个路径非空

2. IU X-Ray 数据约束
   - 图片数量必须为 2（正位 + 侧位）

3. 一致性
   - <image> token 数量 vs images 列表长度
"""

import sys
from typing import List, Tuple

import os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import ImageToReportSample
from report_executor import FormatChecker


class IUXRayFormatChecker(FormatChecker):
    """
    IU X-Ray 数据集格式检查器

    检查项：
    1. sample_id 非空 (error)
    2. instruction 非空 (error)
    3. report 非空 (error)
    4. images 非空列表，每个路径非空 (error)
    5. 图片数量必须为 2（正位 + 侧位）(error)
    6. <image> token 数量 vs images 列表长度 (error)
    """

    EXPECTED_IMAGE_COUNT = 2

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
        # 2. IU X-Ray 数据约束
        # =================================================================

        if sample.images and len(sample.images) != self.EXPECTED_IMAGE_COUNT:
            errors.append(
                f"IU X-Ray 要求 {self.EXPECTED_IMAGE_COUNT} 张图片（正位+侧位），"
                f"实际 {len(sample.images)} 张"
            )

        # =================================================================
        # 3. 一致性
        # =================================================================

        token_count = sample.instruction.count("<image>") if sample.instruction else 0
        image_count = len(sample.images)
        if token_count != image_count:
            errors.append(
                f"<image> token 数量({token_count}) != 图片数量({image_count})"
            )

        return errors, warnings
