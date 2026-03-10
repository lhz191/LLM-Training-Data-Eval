#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX 多轮数据集执行器

包含:
- WebLINXSessionFormatChecker: Session 级别格式检查
- WebLINXSessionStaticChecker: Session 级别静态可执行性检查 (TODO)
- WebLINXSessionDynamicChecker: Session 级别动态可执行性检查 (TODO)

WebLINX 多轮数据特点：
- 一个 Session 包含多轮 Record（按 instructor utterance 切分）
- 每轮 Record 内的 action 类型与单轮相同: click, text_input, say, load, scroll, change, submit
- 跨轮交互：instructor 的新指令触发新轮次
"""

from .constants import (
    UID_REQUIRED_ACTIONS,
    VALUE_REQUIRED_ACTIONS,
    VALID_ACTION_TYPES,
    DEFAULT_VIEWPORT_WIDTH,
    DEFAULT_VIEWPORT_HEIGHT,
)

from .WebLINXSessionFormatChecker import WebLINXSessionFormatChecker
from .WebLINXSessionStaticChecker import WebLINXSessionStaticChecker

__all__ = [
    'WebLINXSessionFormatChecker',
    'WebLINXSessionStaticChecker',
    'UID_REQUIRED_ACTIONS',
    'VALUE_REQUIRED_ACTIONS',
    'VALID_ACTION_TYPES',
    'DEFAULT_VIEWPORT_WIDTH',
    'DEFAULT_VIEWPORT_HEIGHT',
]
