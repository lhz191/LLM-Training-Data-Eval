"""
Image-Text Data Evaluation

评估 Image-Text 数据集的质量，包括：
- Inception Score: 图像质量与多样性
- Prompt Fidelity: 图像-文本对齐度
- Well-Formed Rate: 格式正确率
- C2PA Validation: 内容来源可信度
"""

from .data_types import ImageTextSample
from .loaders import BaseLoader, GeneralLoader
from .image_executor import (
    FormatChecker,
    get_format_checker,
    register_format_checker,
    list_format_checkers,
)

# 导入具体实现以触发注册
from . import coco_executor

__all__ = [
    'ImageTextSample',
    'BaseLoader',
    'GeneralLoader',
    'FormatChecker',
    'get_format_checker',
    'register_format_checker',
    'list_format_checkers',
]
