#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX 数据集执行器

包含:
- WebLINXFormatChecker: 格式检查器

WebLINX 数据特点：
- 数据按 action 分割，需要按 demo 聚合成 Record
- action 类型: click, text_input, say, load, scroll, change, submit
- 某些操作需要 uid (click, text_input, change, submit)
- 某些操作需要 value (text_input, say, load, scroll, change)
- utterances 可能是无效的 "N o   i n s t r u c t o r   u t t e r a n c e"
"""

import re
from typing import Dict, List, Tuple, Optional, Any

from text_gui_executor import (
    FormatChecker,
    HTMLLocator,
    register_format_checker,
    register_html_locator,
)
from loaders import Record, Action


# =============================================================================
# WebLINX Action 类型定义
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
# 格式检查器
# =============================================================================

class WebLINXFormatChecker(FormatChecker):
    """
    WebLINX 数据格式检查器
    
    检查项：
    1. Record 级别
       - demo_id 是否存在
       - utterances 是否有效（非空且非占位符）
       - actions 是否存在且非空
       
    2. Action 级别
       - action_type 是否合法
       - 需要 uid 的操作是否有 target_element
       - 需要 value 的操作是否有 action_value
       - candidates 存在时，uid 是否在其中
       
    3. 数据质量检查
       - clean_html 是否为空
       - viewport 是否为空
    """
    
    def check(self, record: Record) -> Tuple[List[str], List[str]]:
        """检查 WebLINX Record 的数据格式"""
        errors = []
        warnings = []  # 保留接口，但不使用
        
        # === 1. Record 级别检查 ===
        
        # demo_id
        demo_id = record.metadata.get('demo_id', '')
        if not demo_id:
            errors.append("Record missing 'demo_id' in metadata")
        
        # utterances - 不检查，WebLINX 的 utterances 可能为空或是占位符
        # website - 不检查，可能无法从 actions 中提取
        
        # actions
        if not record.actions:
            errors.append("Record has no actions")
            return errors, warnings  # 无法继续检查 action 级别
        
        # === 2. Action 级别检查 ===
        for i, action in enumerate(record.actions):
            action_errors, action_warnings = self._check_action(action, i)
            errors.extend(action_errors)
            # warnings 不再使用
        
        return errors, warnings
    
    def _check_action(self, action: Action, idx: int) -> Tuple[List[str], List[str]]:
        """检查单个 Action 的格式"""
        errors = []
        warnings = []  # 保留接口，但不使用
        prefix = f"Action[{idx}]"
        
        action_type = action.action_type
        action_value = action.action_value
        action_repr = action.action_repr
        target_element = action.target_element  # uid 字符串
        candidates = action.candidates
        
        # === 1. action_type 检查 ===
        if not action_type:
            errors.append(f"{prefix}: missing 'action_type'")
        elif action_type == 'unknown':
            errors.append(f"{prefix}: unknown action type, action_repr='{action_repr[:80] if action_repr else 'N/A'}'")
        elif action_type not in VALID_ACTION_TYPES:
            errors.append(f"{prefix}: invalid action_type '{action_type}', must be one of {VALID_ACTION_TYPES}")
        
        # === 2. target_element (uid) 检查 ===
        # target_element 是 loader 从 action_repr 中提取的 uid
        # 如果提取失败（uid=None 或 uid="" 或没有 uid），target_element 就是 None
        if action_type in UID_REQUIRED_ACTIONS:
            if not target_element:  # None 或空字符串
                errors.append(f"{prefix}: {action_type} missing or invalid uid")
        
        # === 3. action_value 检查 ===
        if action_type in VALUE_REQUIRED_ACTIONS:
            expected_param = VALUE_REQUIRED_ACTIONS[action_type]
            if not action_value:
                errors.append(f"{prefix}: {action_type} missing value (expected '{expected_param}')")
        
        # === 4. candidates 一致性检查 ===
        if target_element and candidates:
            # 检查 uid 是否在 candidates 中
            uid_in_candidates = self._check_uid_in_candidates(target_element, candidates)
            if not uid_in_candidates:
                errors.append(f"{prefix}: target uid not found in candidates")
        
        # === 5. 数据质量检查 ===
        
        # clean_html
        # 注意：WebLINX 的 clean_html 是压缩格式，uid 会被截断（如 a3...9c）
        # 所以不检查 uid 是否在 clean_html 中，这是数据集设计特点而非错误
        if not action.cleaned_html:
            errors.append(f"{prefix}: empty 'clean_html'")
        
        # viewport - 不检查，可能为空
        # utterances - 不检查，可能是占位符
        
        return errors, warnings
    
    def _check_uid_in_candidates(self, target_uid: str, candidates: List[Dict]) -> bool:
        """检查 target uid 是否在 candidates 中"""
        if not target_uid or not candidates:
            return False
        
        for cand in candidates:
            cand_uid = cand.get('uid', '')
            if cand_uid == target_uid:
                return True
        
        return False


# =============================================================================
# HTML 定位器
# =============================================================================

class WebLINXLocator(HTMLLocator):
    """
    WebLINX HTML 定位器
    
    定位方式：通过 data-webtasks-id (uid)
    格式：data-webtasks-id="f1d2b03c-8fc6-445b"
    
    注意：WebLINX 的 clean_html 是压缩格式，uid 值会被截断（如 f1d2...445b）
    所以 clean_html 的定位率预期很低（约 2.4%），而 raw_html 定位率应该更高。
    
    这个指标主要用于计算信息保留率：
    - retention_rate = clean_success / raw_success
    - 如果保留率很低，说明 clean 过程丢失了关键定位信息
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
        
        # target_element 在 WebLINX 中是 uid 字符串
        target_uid = action.target_element
        
        # 如果不需要 uid 的操作，跳过定位检查
        action_type = action.action_type
        if action_type not in UID_REQUIRED_ACTIONS:
            return True, "no_uid_required"
        
        if not target_uid:
            return False, "no_target_uid"
        
        # 检查完整 uid 是否在 html 中
        # 格式可能是: data-webtasks-id="xxx" 或 data-webtasks-id='xxx'
        if target_uid in html:
            return True, "found"
        else:
            return False, "uid_not_found"


# =============================================================================
# 注册检查器和定位器
# =============================================================================

register_format_checker('weblinx', WebLINXFormatChecker)
register_html_locator('weblinx', WebLINXLocator)


# =============================================================================
# 命令行测试
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='WebLINX Format Check 测试')
    parser.add_argument('--data-path', type=str,
                        default='/home/liuhaoze/Desktop/mind2web/weblinx',
                        help='WebLINX 数据目录')
    parser.add_argument('--split', type=str, default='train',
                        help='数据集 split')
    parser.add_argument('--max-samples', type=int, default=10,
                        help='最大样本数')
    args = parser.parse_args()
    
    from loaders import WebLINXLoader
    
    print("=" * 60)
    print("WebLINX Format Check 测试")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"Split: {args.split}")
    print(f"样本数: {args.max_samples}")
    print()
    
    # 加载数据
    loader = WebLINXLoader(args.data_path, args.split)
    records = loader.parse_all(show_progress=True)
    
    print(f"\n加载了 {len(records)} 条记录")
    print()
    
    # 创建检查器
    checker = WebLINXFormatChecker()
    
    # 检查前 N 条
    total_errors = 0
    total_warnings = 0
    records_with_errors = 0
    
    for i, record in enumerate(records[:args.max_samples]):
        errors, warnings = checker.check(record)
        
        demo_id = record.metadata.get('demo_id', 'N/A')
        if errors or warnings:
            print(f"[{i+1}] {demo_id}: {len(errors)} errors, {len(warnings)} warnings")
            for e in errors[:3]:
                print(f"  ❌ {e}")
            for w in warnings[:3]:
                print(f"  ⚠️ {w}")
        else:
            print(f"[{i+1}] {demo_id}: ✅ PASS")
        
        total_errors += len(errors)
        total_warnings += len(warnings)
        if errors:
            records_with_errors += 1
    
    print()
    print("=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"检查记录数: {min(args.max_samples, len(records))}")
    print(f"有错误的记录: {records_with_errors}")
    print(f"总错误数: {total_errors}")
    print(f"总警告数: {total_warnings}")
