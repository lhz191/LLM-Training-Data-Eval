#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX HTML 定位器
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
from .constants import UID_REQUIRED_ACTIONS


class WebLINXLocator(HTMLLocator):
    """
    WebLINX HTML 定位器
    
    定位方式：通过 data-webtasks-id (uid)
    格式：data-webtasks-id="f1d2b03c-8fc6-445b"
    
    支持两种匹配：
    1. 完整匹配：uid 直接在 HTML 中
    2. 截断匹配：HTML 中的 uid 是截断格式（如 f1d2...445b），检查 prefix/suffix
    
    这个指标主要用于计算信息保留率：
    - retention_rate = clean_success / raw_success
    """
    
    def can_locate(self, action, html: str) -> Tuple[bool, str]:
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
        
        # target_element 在 WebLINX 中是 uid 字符串
        target_uid = action.target_element
        
        # 如果不需要 uid 的操作，跳过定位检查
        action_type = action.action_type
        if action_type not in UID_REQUIRED_ACTIONS:
            return True, "no_uid_required"
        
        if not target_uid:
            return False, "no_target_uid"
        
        # 1. 完整匹配：uid 直接在 HTML 中
        if target_uid in html:
            return True, "found_exact"
        
        # 2. 截断匹配：检查 HTML 中截断的 uid 是否与完整 uid 对应
        if self._check_truncated_uid(target_uid, html):
            return True, "found_truncated"
        
        return False, "uid_not_found"
    
    def locate_with_depth(self, action, html: str) -> Tuple[bool, int, str]:
        """
        定位元素并返回 DOM 深度
        
        先调用 can_locate 检查是否能定位，然后用 xpath 计算深度近似。
        
        Args:
            action: Action 对象
            html: HTML 字符串
            
        Returns:
            (success, depth, reason)
            - success: 是否定位成功
            - depth: DOM 深度（xpath 中 '/' 的数量作为近似）
            - reason: 原因说明
        """
        # 先检查是否能定位
        can_locate_result, reason = self.can_locate(action, html)
        if not can_locate_result:
            return False, -1, reason
        
        # 如果不需要 uid 的操作（say, scroll, load 等），跳过深度计算
        action_type = action.action_type
        if action_type not in UID_REQUIRED_ACTIONS:
            return True, -1, "no_uid_required"
        
        # 从 candidates 中获取 xpath 来计算深度
        # （因为 cleaned_html 是非标准格式，无法用 BeautifulSoup 解析）
        if not action.candidates:
            return True, -1, "no_candidates_for_depth"
        
        # 取第一个 candidate（通常是目标元素）
        candidate = action.candidates[0]
        if not isinstance(candidate, dict):
            return True, -1, "invalid_candidate_for_depth"
        
        xpath = candidate.get('xpath', '')
        if not xpath:
            return True, -1, "no_xpath_for_depth"
        
        # 计算 xpath 深度：'/' 的数量作为近似
        # 例如 /html/body/div/span → 深度 4
        depth = xpath.count('/')
        
        return True, depth, "found"
    
    def _check_truncated_uid(self, full_uid: str, html: str) -> bool:
        """
        检查完整 uid 是否与 HTML 中截断的 uid 匹配
        
        截断格式: "前缀...后缀" (如 1...0f 对应 1b010db4-3df2-4c0f)
        """
        # 提取 HTML 中所有的 data-webtasks-id 值
        pattern = r'data-webtasks-id=["\']([^"\']+)["\']'
        uid_values = re.findall(pattern, html)
        
        for uid_val in uid_values:
            if self._match_truncated(full_uid, uid_val):
                return True
        return False
    
    def _match_truncated(self, full_uid: str, truncated_uid: str) -> bool:
        """
        检查完整 uid 是否匹配截断的 uid
        
        匹配规则: full_uid.startswith(前缀) and full_uid.endswith(后缀)
        """
        if not full_uid or not truncated_uid:
            return False
        
        # 完全匹配
        if full_uid == truncated_uid:
            return True
        
        # 截断匹配
        if '...' in truncated_uid:
            parts = truncated_uid.split('...')
            if len(parts) == 2:
                prefix, suffix = parts
                return full_uid.startswith(prefix) and full_uid.endswith(suffix)
        
        return False
