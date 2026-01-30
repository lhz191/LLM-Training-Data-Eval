#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX HTML 定位器
"""

import re
from typing import Tuple

from .constants import UID_REQUIRED_ACTIONS


class WebLINXLocator:
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
