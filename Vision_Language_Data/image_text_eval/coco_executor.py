#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO Caption 数据集执行器

实现 COCO Caption 数据集的格式检查器。
"""

import os
import re

from data_types import ImageTextSample
from image_executor import FormatChecker, register_format_checker


# 用于解析 image_path 中的 image_id
_COCO_PATH_RE = re.compile(r"COCO_train2014_0*(\d+)\.jpg$")


@register_format_checker("coco_caption")
class COCOCaptionFormatChecker(FormatChecker):
    """COCO Caption 数据集格式检查器"""
    
    def __init__(self, strict: bool = True, dataset_root: str = None):
        """
        Args:
            strict: 是否使用严格模式
            dataset_root: 数据集根目录（用于验证文件存在性）
        """
        self.strict = strict
        self.dataset_root = dataset_root
    
    def check(self, sample: ImageTextSample) -> bool:
        """验证 COCO Caption 数据项"""
        
        # 1) 必需字段是否齐全
        if not sample.image_path or not sample.caption:
            return False

        # 2) 严格模式下的基础值检查
        if self.strict:
            if sample.sample_id is not None and sample.sample_id <= 0:
                return False
            if sample.image_id is not None and sample.image_id <= 0:
                return False
            if not sample.caption.strip():
                return False

        # 3) image_path 格式检查
        path = sample.image_path.replace("\\", "/")
        match = _COCO_PATH_RE.search(path)
        
        if self.strict and match is None:
            return False

        # 4) 正确绑定：image_id 必须与路径中的 ID 一致
        if match is not None and sample.image_id is not None:
            if int(match.group(1)) != sample.image_id:
                return False

        # 5) grounded：图片文件必须真实存在
        if self.dataset_root is not None:
            if not os.path.exists(sample.image_path):
                return False

        return True
