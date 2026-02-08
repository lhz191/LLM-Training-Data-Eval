#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebShop HTML 定位器
"""

import os
import sys
import re
from typing import Tuple

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from text_gui_executor import HTMLLocator
from data_types import Action


class WebShopLocator(HTMLLocator):
    """
    WebShop HTML 定位器
    
    WebShop 的 state（作为 cleaned_html）是纯文本格式，包含按钮信息：
    - [button] xxx [button_]: 可点击按钮
    - [clicked button] xxx [clicked button_]: 已点击按钮
    - [Search]: 搜索按钮
    
    定位方式：检查 action 的 target 是否在 state 文本中
    """
    
    def can_locate(self, action: Action, html: str) -> Tuple[bool, str]:
        """
        检查是否能在 HTML（state 文本）中定位到 target
        
        Args:
            action: Action 对象
            html: HTML 字符串（实际是 WebShop 的 state 文本）
            
        Returns:
            (success, reason)
        """
        if not html:
            return False, "empty_html"
        
        action_type = action.action_type
        action_repr = action.metadata.get('action_repr', '')
        
        if not action_repr:
            return False, "no_action_repr"
        
        html_lower = html.lower()
        
        if action_type == 'search':
            # search: 检查 [Search] 按钮是否存在
            if '[search]' in html_lower:
                return True, "found"
            else:
                return False, "search_button_not_found"
        
        elif action_type == 'click':
            # click: 提取 target，检查 [button] xxx [button_] 或 [clicked button] xxx [clicked button_]
            if action_repr.startswith('click[') and action_repr.endswith(']'):
                target = action_repr[6:-1].lower()
                pattern1 = f'[button] {target} [button_]'
                pattern2 = f'[clicked button] {target} [clicked button_]'
                
                if pattern1 in html_lower or pattern2 in html_lower:
                    return True, "found"
                else:
                    return False, "target_not_found"
            else:
                return False, "invalid_action_format"
        
        else:
            return False, f"unknown_action_type_{action_type}"
    
    def locate_with_depth(self, action: Action, html: str) -> tuple:
        """
        定位元素并返回 DOM 深度
        
        WebShop 的 state 是纯文本格式（[button] xxx [button_]），
        不是真正的 HTML DOM 结构。信息直接展示，没有嵌套，
        因此默认认为是"浮于表面"，返回深度 1。
        
        Args:
            action: Action 对象
            html: state 文本
            
        Returns:
            (success, depth, reason)
            - depth 返回 1，表示信息浮于表面
        """
        can_locate_result, reason = self.can_locate(action, html)
        
        if not can_locate_result:
            return False, -1, reason
        
        # WebShop 是文本格式，信息直接暴露，默认浮于表面
        return True, 1, "text_format_surface"