#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX 常量定义（多轮版，与单轮一致）
"""

# =============================================================================
# Action 类型定义
# =============================================================================

UID_REQUIRED_ACTIONS = {'click', 'text_input', 'change', 'submit'}

VALUE_REQUIRED_ACTIONS = {
    'text_input': 'text',
    'say': 'utterance',
    'load': 'url',
    'scroll': 'xy',
    'change': 'value',
}

VALID_ACTION_TYPES = {'click', 'text_input', 'say', 'load', 'scroll', 'change', 'submit'}


# =============================================================================
# 视口默认值
# =============================================================================

DEFAULT_VIEWPORT_WIDTH = 1536
DEFAULT_VIEWPORT_HEIGHT = 714
