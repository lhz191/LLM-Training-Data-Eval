#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image-to-Report 执行器 - 基础模块

继承 common.base 中的统一基类，定义 image-to-report 模态的检查器接口。

继承链：
  common.base.BaseFormatChecker
    └── report_executor.FormatChecker  (modality='image_to_report')
          ├── GeneralFormatChecker     (通用，不依赖数据集)
          ├── IUXRayFormatChecker      (IU X-Ray 特有)
          └── ShareGPT4VFormatChecker  (ShareGPT4V 特有)

  common.base.BaseQualityChecker
    └── report_executor.QualityChecker  (modality='image_to_report')
          ├── IUXRayQualityChecker     (医学术语、报告结构)
          └── ShareGPT4VQualityChecker (描述质量、细节丰富度)

使用方式:
    from report_executor import FormatChecker, QualityChecker
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Tuple, Dict, Any
from abc import abstractmethod

from data_types import ImageToReportSample

from common.base import BaseFormatChecker, BaseQualityChecker


# =============================================================================
# 基类
# =============================================================================

class FormatChecker(BaseFormatChecker):
    """
    格式检查器基类（Image-to-Report）

    继承自 common.base.BaseFormatChecker

    每个数据集实现自己的 check 方法，返回 (errors, warnings) 元组。
    """

    @property
    def modality(self) -> str:
        return 'image_to_report'

    @abstractmethod
    def check(self, sample: ImageToReportSample) -> Tuple[List[str], List[str]]:
        """
        检查样本的格式正确性

        Args:
            sample: ImageToReportSample 样本

        Returns:
            (errors, warnings) 元组
            - errors: 错误列表（严重问题，影响数据可用性）
            - warnings: 警告列表（轻微问题，不影响数据可用性）
        """
        pass


class QualityChecker(BaseQualityChecker):
    """
    报告质量评估器基类（Image-to-Report）

    继承自 common.base.BaseQualityChecker

    每个数据集实现自己的 check 方法，评估方式不限：
    - LLM Judge（文本模型即可，图文相关性在 validity 里已做）
    - 规则匹配
    - 外部工具

    返回 Dict 包含各维度评分和详情。
    """

    @property
    def modality(self) -> str:
        return 'image_to_report'

    @abstractmethod
    def check(self, sample: ImageToReportSample) -> Dict[str, Any]:
        """
        评估样本的报告质量

        Args:
            sample: ImageToReportSample 样本

        Returns:
            评估结果字典，至少包含:
            - scores: Dict[str, float], 各维度评分
            - passed: bool, 是否通过质量门槛
        """
        pass
