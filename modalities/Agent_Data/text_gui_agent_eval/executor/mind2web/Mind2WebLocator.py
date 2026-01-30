#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mind2Web HTML 定位器
"""

import os
import sys
from typing import Tuple

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
