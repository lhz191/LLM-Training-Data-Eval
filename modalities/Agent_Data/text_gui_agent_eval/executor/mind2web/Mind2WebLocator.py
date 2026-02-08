#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mind2Web HTML 定位器
"""

import os
import sys
from typing import Tuple

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from text_gui_executor import HTMLLocator
from data_types import Action


class Mind2WebLocator(HTMLLocator):
    """
    Mind2Web HTML 定位器
    
    定位方式：通过 backend_node_id
    格式：<tag backend_node_id="136" ...>
    
    Mind2Web 的 cleaned_html 保留了 backend_node_id 属性，
    所以理论上定位率应该很高。
    """
    
    def can_locate(self, action: Action, html: str) -> Tuple[bool, str]:
        """
        检查是否能在 HTML 中定位到 target
        
        Args:
            action: Action 对象
            html: HTML 字符串（可以是 raw_html 或 cleaned_html）
            
        Returns:
            (success, reason)
        """
        if not html:
            return False, "empty_html"
        
        target = action.target_element
        if not target:
            return False, "no_target_element"
        
        backend_node_id = target.get('backend_node_id')
        if not backend_node_id:
            return False, "no_backend_node_id"
        
        node_id_str = str(backend_node_id)
        if node_id_str in html:
            return True, "found"
        else:
            return False, "not_found"
    
    def locate_with_depth(self, action: Action, html: str) -> Tuple[bool, int, str]:
        """
        定位元素并返回 DOM 深度
        
        通过 backend_node_id 定位元素，然后计算其在 DOM 树中的深度。
        
        Args:
            action: Action 对象
            html: HTML 字符串
            
        Returns:
            (success, depth, reason)
            - success: 是否定位成功
            - depth: DOM 深度（从根到目标的层数）
            - reason: 原因说明
        """
        if BeautifulSoup is None:
            return False, -1, "beautifulsoup_not_installed"
        
        if not html:
            return False, -1, "empty_html"
        
        target = action.target_element
        if not target:
            return False, -1, "no_target_element"
        
        backend_node_id = target.get('backend_node_id')
        if not backend_node_id:
            return False, -1, "no_backend_node_id"
        
        node_id_str = str(backend_node_id)
        
        try:
            # 解析 HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # 通过 backend_node_id 属性定位元素
            element = soup.find(attrs={'backend_node_id': node_id_str})
            
            if not element:
                return False, -1, "not_found"
            
            # 计算 DOM 深度：遍历所有 parent 直到根
            depth = 0
            parent = element.parent
            while parent:
                depth += 1
                parent = parent.parent
            
            return True, depth, "found"
            
        except Exception as e:
            return False, -1, f"parse_error: {e}"
