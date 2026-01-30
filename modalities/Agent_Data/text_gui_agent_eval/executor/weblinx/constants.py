#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX 常量定义
"""

# =============================================================================
# Action 类型定义
# =============================================================================

# 需要 uid (target_element) 的操作
UID_REQUIRED_ACTIONS = {'click', 'text_input', 'change', 'submit'}

# 需要 value 的操作
VALUE_REQUIRED_ACTIONS = {
    'text_input': 'text',      # text_input(uid="...", text="...")
    'say': 'utterance',        # say(speaker="...", utterance="...")
    'load': 'url',             # load(url="...")
    'scroll': 'xy',            # scroll(x=..., y=...)
    'change': 'value',         # change(uid="...", value="...")
}

# 所有有效的 action 类型
VALID_ACTION_TYPES = {'click', 'text_input', 'say', 'load', 'scroll', 'change', 'submit'}


# =============================================================================
# 视口默认值
# =============================================================================

# 默认视口大小（如果 action 中没有 viewport 字段）
DEFAULT_VIEWPORT_WIDTH = 1536
DEFAULT_VIEWPORT_HEIGHT = 714


# =============================================================================
# WebLINX 元素属性统计（基于 31,576 个样例）
# =============================================================================
#
# element 顶层字段（全部 100%）：
#   tagName, bbox, attributes, textContent, xpath, outerHTML, innerHTML
#
# attributes 中的关键字段（用于定位和验证）：
#   data-webtasks-id: 98.3%   # UID
#   class:            78.1%   # CSS 类
#   id:               20.1%   # 元素 ID
#   type:             16.7%   # input/button 类型
#   placeholder:      12.3%   # 输入框占位符
#   aria-label:       11.9%   # 无障碍标签
#   tabindex:         11.6%   # Tab 索引
#   autocomplete:      9.1%   # 自动完成
#   value:             8.8%   # 表单元素值
#   name:              8.6%   # 表单元素名
#   role:              8.5%   # ARIA 角色
#   href:              7.2%   # 链接地址
#   title:             2.4%   # 元素标题
#   src:               2.0%   # 图片/视频源
#   alt:               1.8%   # 图片替代文本
#
