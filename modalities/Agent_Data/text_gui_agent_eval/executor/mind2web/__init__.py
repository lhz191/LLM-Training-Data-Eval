#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mind2Web 数据集执行器

包含:
- Mind2WebStaticChecker: 静态可执行性检查器
- Mind2WebDynamicChecker: 动态可执行性检查器
- Mind2WebFormatChecker: 格式检查器
- Mind2WebLocator: HTML 定位器

Mind2Web 数据特点：
- 基于 MHTML 快照验证
- 使用坐标定位和属性定位两种方式
"""

# 导入检查器类
from .Mind2WebStaticChecker import Mind2WebStaticChecker
from .Mind2WebDynamicChecker import Mind2WebDynamicChecker
from .Mind2WebFormatChecker import Mind2WebFormatChecker
from .Mind2WebLocator import Mind2WebLocator

# 导入常量
from .constants import VIEWPORT_WIDTH, VIEWPORT_HEIGHT

# 导入工具函数
from .utils import (
    is_dynamic_class,
    escape_css_value,
    parse_candidate,
    find_element_by_all_attributes,
    verify_by_coords,
    verify_by_attrs,
    verify_element_match,
)

__all__ = [
    # 检查器
    'Mind2WebStaticChecker',
    'Mind2WebDynamicChecker',
    'Mind2WebFormatChecker',
    'Mind2WebLocator',
    # 常量
    'VIEWPORT_WIDTH',
    'VIEWPORT_HEIGHT',
    # 工具函数
    'is_dynamic_class',
    'escape_css_value',
    'parse_candidate',
    'find_element_by_all_attributes',
    'verify_by_coords',
    'verify_by_attrs',
    'verify_element_match',
]
