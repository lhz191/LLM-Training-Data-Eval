#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX 数据集执行器

包含:
- WebLINXStaticChecker: 静态可执行性检查器
- WebLINXFormatChecker: 格式检查器
- WebLINXLocator: HTML 定位器

WebLINX 数据特点：
- 数据按 action 分割，需要按 demo 聚合成 Record
- action 类型: click, text_input, say, load, scroll, change, submit
- 某些操作需要 uid (click, text_input, change, submit)
- 某些操作需要 value (text_input, say, load, scroll, change)
"""

# 导入检查器类
from .WebLINXStaticChecker import WebLINXStaticChecker
from .WebLINXFormatChecker import WebLINXFormatChecker
from .WebLINXLocator import WebLINXLocator

# 导入常量
from .constants import (
    UID_REQUIRED_ACTIONS,
    VALUE_REQUIRED_ACTIONS,
    VALID_ACTION_TYPES,
    DEFAULT_VIEWPORT_WIDTH,
    DEFAULT_VIEWPORT_HEIGHT,
)

# 导入工具函数
from .utils import (
    is_dynamic_class,
    escape_css_value,
    parse_weblinx_candidate,
    find_candidate_by_uid,
    build_css_selector,
    truncated_match,
    verify_weblinx_element_match,
)

__all__ = [
    # 检查器
    'WebLINXStaticChecker',
    'WebLINXFormatChecker',
    'WebLINXLocator',
    # 常量
    'UID_REQUIRED_ACTIONS',
    'VALUE_REQUIRED_ACTIONS',
    'VALID_ACTION_TYPES',
    'DEFAULT_VIEWPORT_WIDTH',
    'DEFAULT_VIEWPORT_HEIGHT',
    # 工具函数
    'is_dynamic_class',
    'escape_css_value',
    'parse_weblinx_candidate',
    'find_candidate_by_uid',
    'build_css_selector',
    'truncated_match',
    'verify_weblinx_element_match',
]
