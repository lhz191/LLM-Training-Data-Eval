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
