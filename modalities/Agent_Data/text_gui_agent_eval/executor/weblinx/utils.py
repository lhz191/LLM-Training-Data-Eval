#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX 工具函数

包含：
- CSS 相关：is_dynamic_class, escape_css_value, build_css_selector
- 数据解析：parse_weblinx_candidate, find_candidate_by_uid
- 元素验证：verify_weblinx_element_match, truncated_match
"""

import re
from typing import Dict, List, Tuple, Optional, Any


# =============================================================================
# CSS 工具函数
# =============================================================================

def is_dynamic_class(c: str) -> bool:
    """判断是否是动态生成的 class（CSS-in-JS 等）或无效的 CSS 类名"""
    if not c:
        return True
    return (
        c.startswith('css-') or            # Emotion/styled-components
        c.startswith('jss') or             # JSS
        c.startswith('_') or               # Angular/Vue 等框架生成
        c.startswith('ng-') or             # Angular
        c.startswith('sc-') or             # Styled-components
        (len(c) > 0 and c[0].isdigit()) or # 数字开头
        (len(c) <= 10 and any(ch.isdigit() for ch in c)) or  # 短且含数字
        ':' in c or                        # Tailwind 变体类 (hover:xxx)
        '[' in c or ']' in c               # Tailwind 任意值类 ([color:red])
    )


def escape_css_value(s: str) -> str:
    """转义 CSS 属性值中的特殊字符"""
    if not s:
        return s
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace("'", "\\'")
    s = s.replace('\n', ' ')
    s = s.replace('\r', ' ')
    return s


# =============================================================================
# 数据解析函数
# =============================================================================

def parse_weblinx_candidate(candidate: dict) -> dict:
    """
    解析 WebLINX candidates 中的元素信息（训练数据）
    
    Candidate 结构（来自 train.json.gz，loader 解析后）：
    {
        'uid': '9f2c37b3-a223-4f07',     # 100.0% 覆盖
        'tag': 'body',                    # 100.0% 覆盖
        'xpath': '/html/body',            # 99.8% 覆盖
        'text': '',                       # 47.6% 有内容
        'bbox': 'x=0 y=0 width=1519.2 height=2919.5',  # 100.0% 覆盖，字符串格式
        'attributes': {                   # 99.9% 覆盖，已解析的字典
            'class': '...',               # 19.4%，可能被截断
            'data-webtasks-id': '...',    # 10.0%
            'type': '...',                # 4.4%
            'id': '...',                  # 2.5%
            'role': '...',                # 2.3%
            'href': '...',                # 1.3%
            'aria-label': '...',          # 1.0%
            'name': '...',                # 1.0%
            'title': '...',               # 0.8%
            'placeholder': '...',         # 0.6%
            'alt': '...',                 # 0.5%
            'value': '...',               # 2.5%
            'tabindex': '...',            # 6.7%
            'aria-hidden': '...',         # 2.0%
            'aria-expanded': '...',       # 0.8%
            'target': '...',              # 1.2%
            'data-testid': '...',         # 0.8%
            ...
        },
        'children': 'div span',           # 47.3% 有内容
    }
    
    注意：attributes 中的值可能被截断（包含 ...）
    """
    if not candidate:
        return {}
    
    # 如果是字符串，无法解析，返回空
    if isinstance(candidate, str):
        return {}
    
    result = {
        'tag': (candidate.get('tag') or '').lower(),
        'bbox': None,
        'xpath': candidate.get('xpath'),
        
        # CSS 可筛选的属性（按出现频率排序）
        'class': None,           # 19.4%
        'id': None,              # 2.5%
        'type': None,            # 4.4%
        'role': None,            # 2.3%
        'name': None,            # 1.0%
        'href': None,            # 1.3%
        'aria_label': None,      # 1.0%
        'aria_hidden': None,     # 2.0%
        'aria_expanded': None,   # 0.8%
        'placeholder': None,     # 0.6%
        'title': None,           # 0.8%
        'alt': None,             # 0.5%
        'value': None,           # 2.5%
        'target': None,          # 1.2%
        'tabindex': None,        # 6.7%
        'data_testid': None,     # 0.8%
        
        # 文本内容
        'text_content': candidate.get('text', ''),
        
        # UID
        'data_webtasks_id': candidate.get('uid'),
    }
    
    # 解析 bbox 字符串 "x=0 y=0 width=1519.2 height=2919.5"
    bbox_str = candidate.get('bbox', '')
    if bbox_str:
        x_match = re.search(r'x=([0-9.-]+)', bbox_str)
        y_match = re.search(r'y=([0-9.-]+)', bbox_str)
        w_match = re.search(r'width=([0-9.-]+)', bbox_str)
        h_match = re.search(r'height=([0-9.-]+)', bbox_str)
        if x_match and y_match and w_match and h_match:
            result['bbox'] = {
                'x': float(x_match.group(1)),
                'y': float(y_match.group(1)),
                'width': float(w_match.group(1)),
                'height': float(h_match.group(1)),
            }
    
    # 提取 attributes（已经是字典格式）
    attrs = candidate.get('attributes', {})
    if isinstance(attrs, dict):
        result['class'] = attrs.get('class', '')
        result['id'] = attrs.get('id', '')
        result['type'] = attrs.get('type', '')
        result['role'] = attrs.get('role', '')
        result['name'] = attrs.get('name', '')
        result['href'] = attrs.get('href', '')
        result['aria_label'] = attrs.get('aria-label', '')
        result['aria_hidden'] = attrs.get('aria-hidden', '')
        result['aria_expanded'] = attrs.get('aria-expanded', '')
        result['placeholder'] = attrs.get('placeholder', '')
        result['title'] = attrs.get('title', '')
        result['alt'] = attrs.get('alt', '')
        result['value'] = attrs.get('value', '')
        result['target'] = attrs.get('target', '')
        result['tabindex'] = attrs.get('tabindex', '')
        result['data_testid'] = attrs.get('data-testid', '')
        # UID 优先从 attributes 获取（如果存在）
        result['data_webtasks_id'] = attrs.get('data-webtasks-id', candidate.get('uid'))
    
    return result


def find_candidate_by_uid(target_uid: str, candidates: list) -> Optional[dict]:
    """
    根据 uid 在 candidates 中找到目标元素
    
    Args:
        target_uid: 目标元素的 UID
        candidates: candidates 列表（可以是字符串列表或已解析的字典列表）
        
    Returns:
        解析后的元素信息字典，或 None
    """
    if not candidates or not target_uid:
        return None
    
    for cand in candidates:
        # 如果是字符串，跳过（无法解析）
        if isinstance(cand, str):
            continue
        
        # 检查 uid（loader 格式）或 data_webtasks_id（parse_weblinx_candidate 格式）
        cand_uid = cand.get('uid') or cand.get('data_webtasks_id')
        if cand_uid == target_uid:
            # 返回 parse_weblinx_candidate 格式（统一格式）
            return parse_weblinx_candidate(cand)
    
    return None


# =============================================================================
# CSS 选择器构建
# =============================================================================

def build_css_selector(info: dict) -> Tuple[str, str]:
    """
    根据元素信息构建 CSS 选择器
    
    Args:
        info: 解析后的元素信息（来自 parse_weblinx_candidate）
        
    Returns:
        (selector, description)
    """
    # 按 parse_weblinx_candidate 的顺序提取字段
    tag = info.get('tag', '')
    cls = info.get('class', '')           # 1. class (19.4%)
    elem_id = info.get('id', '')          # 2. id (2.5%)
    elem_type = info.get('type', '')      # 3. type (4.4%)
    role = info.get('role', '')           # 4. role (2.3%)
    name = info.get('name', '')           # 5. name (1.0%)
    href = info.get('href', '')           # 6. href (1.3%)
    aria_label = info.get('aria_label', '')      # 7. aria_label (1.0%)
    aria_hidden = info.get('aria_hidden', '')    # 8. aria_hidden (2.0%)
    aria_expanded = info.get('aria_expanded', '') # 9. aria_expanded (0.8%)
    placeholder = info.get('placeholder', '')    # 10. placeholder (0.6%)
    title = info.get('title', '')         # 11. title (0.8%)
    alt = info.get('alt', '')             # 12. alt (0.5%)
    value = info.get('value', '')         # 13. value (2.5%)
    target = info.get('target', '')       # 14. target (1.2%)
    tabindex = info.get('tabindex', '')   # 15. tabindex (6.7%)
    data_testid = info.get('data_testid', '')  # 16. data_testid (0.8%)
    
    selector_parts = []
    conditions_desc = []
    
    # 按 parse_weblinx_candidate 的顺序构建选择器
    # 0. tag（始终第一）
    if tag:
        selector_parts.append(tag)
        conditions_desc.append(f"tag={tag}")
    
    # 1. class（过滤动态 class，处理截断 class）
    if cls:
        class_count = 0
        for c in cls.split():
            if not c or is_dynamic_class(c):
                continue
            
            if '...' in c:
                # 截断的 class：提取前缀，用包含匹配 [class*="prefix"]
                prefix = c.split('...')[0]
                if prefix and not is_dynamic_class(prefix):
                    selector_parts.append(f'[class*="{escape_css_value(prefix)}"]')
                    class_count += 1
            else:
                # 完整的 class：用精确匹配 .classname
                selector_parts.append(f'.{c}')
                class_count += 1
        
        if class_count > 0:
            conditions_desc.append(f"class({class_count}个)")
    
    # 2. id
    if elem_id:
        selector_parts.append(f'[id="{escape_css_value(elem_id)}"]')
        conditions_desc.append("id")
    
    # 3. type
    if elem_type:
        selector_parts.append(f'[type="{escape_css_value(elem_type)}"]')
        conditions_desc.append("type")
    
    # 4. role
    if role:
        selector_parts.append(f'[role="{escape_css_value(role)}"]')
        conditions_desc.append("role")
    
    # 5. name
    if name:
        selector_parts.append(f'[name="{escape_css_value(name)}"]')
        conditions_desc.append("name")
    
    # 6. href（不截断，用数据集提供的完整值）
    if href:
        selector_parts.append(f'[href="{escape_css_value(href)}"]')
        conditions_desc.append("href")
    
    # 7. aria-label
    if aria_label:
        selector_parts.append(f'[aria-label="{escape_css_value(aria_label)}"]')
        conditions_desc.append("aria-label")
    
    # 8. aria-hidden
    if aria_hidden:
        selector_parts.append(f'[aria-hidden="{escape_css_value(aria_hidden)}"]')
        conditions_desc.append("aria-hidden")
    
    # 9. aria-expanded
    if aria_expanded:
        selector_parts.append(f'[aria-expanded="{escape_css_value(aria_expanded)}"]')
        conditions_desc.append("aria-expanded")
    
    # 10. placeholder
    if placeholder:
        selector_parts.append(f'[placeholder="{escape_css_value(placeholder)}"]')
        conditions_desc.append("placeholder")
    
    # 11. title
    if title:
        selector_parts.append(f'[title="{escape_css_value(title)}"]')
        conditions_desc.append("title")
    
    # 12. alt
    if alt:
        selector_parts.append(f'[alt="{escape_css_value(alt)}"]')
        conditions_desc.append("alt")
    
    # 13. value
    if value:
        selector_parts.append(f'[value="{escape_css_value(value)}"]')
        conditions_desc.append("value")
    
    # 14. target
    if target:
        selector_parts.append(f'[target="{escape_css_value(target)}"]')
        conditions_desc.append("target")
    
    # 15. tabindex
    if tabindex:
        selector_parts.append(f'[tabindex="{escape_css_value(tabindex)}"]')
        conditions_desc.append("tabindex")
    
    # 16. data-testid
    if data_testid:
        selector_parts.append(f'[data-testid="{escape_css_value(data_testid)}"]')
        conditions_desc.append("data-testid")
    
    selector = ''.join(selector_parts)
    desc = '+'.join(conditions_desc)
    
    return selector, desc


# =============================================================================
# 字符串匹配（支持截断）
# =============================================================================

def truncated_match(expected: str, actual: str) -> bool:
    """
    判断可能被截断的 expected 是否匹配完整的 actual
    
    WebLINX 数据集的截断发生在整个 candidates 字符串级别，可能跨属性边界：
    - 原始: [[class]] NITa-value [[attributes]] data-webtasks-id='0cfe742a-535e-4d63'
    - 截断: [[class]] NITa-value...e-4d63
    这里后缀 "e-4d63" 其实来自另一个属性（data-webtasks-id），是截断污染！
    
    匹配策略（对于 prefix...suffix 格式）：
    - 前缀必须在 actual 中存在（前缀是当前属性的真实内容）
    - 后缀不强制要求（后缀可能是从其他属性截断进来的污染）
    """
    if not expected or not actual:
        return expected == actual
    if expected == actual:
        return True
    
    if '...' in expected:
        # 提取前缀（... 之前的部分）
        prefix = expected.split('...')[0]
        
        if prefix:
            # 前缀匹配：actual 等于前缀、以前缀开头、或包含前缀
            if actual == prefix or actual.startswith(prefix) or prefix in actual:
                return True
        
        # 如果没有有效前缀（如 "...suffix"），尝试完整正则
        pattern = re.escape(expected).replace(r'\.\.\.', '.*')
        if re.search(pattern, actual):
            return True
    
    # 子串匹配
    return expected in actual


# =============================================================================
# 元素验证
# =============================================================================

def verify_weblinx_element_match(page, element_handle, expected_info: dict, skip_xpath: bool = False) -> Tuple[bool, str, int, int]:
    """
    验证找到的元素是否与数据集描述匹配
    
    Args:
        page: Playwright page 对象
        element_handle: Playwright 元素句柄
        expected_info: 期望的元素信息（来自 parse_weblinx_candidate）
        skip_xpath: 是否跳过 xpath 验证（坐标定位时使用）
        
    Returns:
        (is_match, reason, matched_count, total_count)
    """
    if not element_handle:
        return False, "element_is_none", 0, 0
    
    # 获取实际元素属性（包括 xpath 和 text）
    try:
        actual_attrs = page.evaluate("""(element) => {
            if (!element || !element.tagName) {
                return null;
            }
            
            // 生成 xpath（使用完整路径格式，与数据集一致）
            function getXPath(el) {
                if (!el) return '';
                // 不使用 id 简写，保持完整路径格式
                if (el === document.body) return '/html/body';
                if (el === document.documentElement) return '/html';
                
                let ix = 0;
                const siblings = el.parentNode ? el.parentNode.childNodes : [];
                for (let i = 0; i < siblings.length; i++) {
                    const sibling = siblings[i];
                    if (sibling === el) {
                        const parentPath = getXPath(el.parentNode);
                        const tag = el.tagName.toLowerCase();
                        // 只有多个同名兄弟时才加索引，与数据集格式一致
                        let sameTagCount = 0;
                        for (let j = 0; j < siblings.length; j++) {
                            if (siblings[j].nodeType === 1 && siblings[j].tagName === el.tagName) {
                                sameTagCount++;
                            }
                        }
                        if (sameTagCount > 1) {
                            return parentPath + '/' + tag + '[' + (ix + 1) + ']';
                        } else {
                            return parentPath + '/' + tag;
                        }
                    }
                    if (sibling.nodeType === 1 && sibling.tagName === el.tagName) {
                        ix++;
                    }
                }
                return '';
            }
            
            return {
                // 按 parse_weblinx_candidate 顺序
                tag: element.tagName.toLowerCase(),
                class: element.getAttribute('class') || '',
                id: element.getAttribute('id') || '',
                type: element.getAttribute('type') || '',
                role: element.getAttribute('role') || '',
                name: element.getAttribute('name') || '',
                href: element.getAttribute('href') || '',
                ariaLabel: element.getAttribute('aria-label') || '',
                ariaHidden: element.getAttribute('aria-hidden') || '',
                ariaExpanded: element.getAttribute('aria-expanded') || '',
                placeholder: element.getAttribute('placeholder') || '',
                title: element.getAttribute('title') || '',
                alt: element.getAttribute('alt') || '',
                value: element.getAttribute('value') || '',
                target: element.getAttribute('target') || '',
                tabindex: element.getAttribute('tabindex') || '',
                dataTestid: element.getAttribute('data-testid') || '',
                xpath: getXPath(element),
                text: (element.textContent || '').trim(),
            };
        }""", element_handle)
        
        if actual_attrs is None:
            return False, "element_stale", 0, 0
    except Exception as e:
        return False, f"evaluate_error: {str(e)[:50]}", 0, 0
    
    # 比较属性（按 parse_weblinx_candidate 顺序）
    mismatches = []
    matches = []
    matched_count = 0
    total_count = 0
    
    # 辅助函数：简单字符串属性验证（支持截断匹配）
    def verify_simple_attr(attr_name: str, expected_key: str, actual_key: str):
        nonlocal matched_count, total_count
        expected_val = expected_info.get(expected_key, '')
        if expected_val:
            total_count += 1
            actual_val = actual_attrs.get(actual_key, '')
            
            if truncated_match(expected_val, actual_val):
                matched_count += 1
                matches.append(f"{attr_name}: ✓")
            else:
                # 截断长字符串避免刷屏
                exp_short = expected_val[:50] + '...' if len(expected_val) > 50 else expected_val
                act_short = (actual_val[:50] + '...') if actual_val and len(actual_val) > 50 else (actual_val or '无')
                mismatches.append(f"{attr_name}: 期望 '{exp_short}', 实际 '{act_short}' ✗")
    
    # 0. 验证 tag
    expected_tag = expected_info.get('tag', '').lower()
    if expected_tag:
        total_count += 1
        if actual_attrs['tag'] == expected_tag:
            matched_count += 1
            matches.append(f"tag: {actual_attrs['tag']} ✓")
        else:
            mismatches.append(f"tag: 期望 '{expected_tag}', 实际 '{actual_attrs['tag']}' ✗")
    
    # 1. 验证 class（过滤动态 class，支持截断匹配）
    expected_class = expected_info.get('class', '')
    if expected_class:
        expected_classes = [c for c in expected_class.split() if c and not is_dynamic_class(c)]
        actual_classes = actual_attrs['class'].split()
        
        if expected_classes:
            total_count += 1
            # 对每个期望的 class，检查是否在实际 classes 中存在匹配
            # truncated_match 已内置前缀匹配逻辑，处理截断污染问题
            missing_classes = []
            for exp_cls in expected_classes:
                found = any(truncated_match(exp_cls, act_cls) for act_cls in actual_classes)
                if not found:
                    missing_classes.append(exp_cls)
            
            if not missing_classes:
                matched_count += 1
                matches.append(f"class: 全部匹配 ({len(expected_classes)}个) ✓")
            else:
                mismatches.append(f"class: 缺少 {missing_classes} ✗")
    
    # 2. 验证 id
    verify_simple_attr('id', 'id', 'id')
    
    # 3. 验证 type
    verify_simple_attr('type', 'type', 'type')
    
    # 4. 验证 role
    verify_simple_attr('role', 'role', 'role')
    
    # 5. 验证 name
    verify_simple_attr('name', 'name', 'name')
    
    # 6. 验证 href
    verify_simple_attr('href', 'href', 'href')
    
    # 7. 验证 aria-label
    verify_simple_attr('aria-label', 'aria_label', 'ariaLabel')
    
    # 8. 验证 aria-hidden
    verify_simple_attr('aria-hidden', 'aria_hidden', 'ariaHidden')
    
    # 9. 验证 aria-expanded
    verify_simple_attr('aria-expanded', 'aria_expanded', 'ariaExpanded')
    
    # 10. 验证 placeholder
    verify_simple_attr('placeholder', 'placeholder', 'placeholder')
    
    # 11. 验证 title
    verify_simple_attr('title', 'title', 'title')
    
    # 12. 验证 alt
    verify_simple_attr('alt', 'alt', 'alt')
    
    # 13. 验证 value
    verify_simple_attr('value', 'value', 'value')
    
    # 14. 验证 target
    verify_simple_attr('target', 'target', 'target')
    
    # 15. 验证 tabindex
    verify_simple_attr('tabindex', 'tabindex', 'tabindex')
    
    # 16. 验证 data-testid
    verify_simple_attr('data-testid', 'data_testid', 'dataTestid')
    
    # 17. 验证 xpath（可能被截断）- 坐标定位时跳过
    expected_xpath = expected_info.get('xpath', '')
    if expected_xpath and not skip_xpath:
        total_count += 1
        actual_xpath = actual_attrs.get('xpath', '')
        if truncated_match(expected_xpath, actual_xpath):
            matched_count += 1
            matches.append(f"xpath: ✓")
        else:
            # 截断长 xpath 避免刷屏
            exp_short = expected_xpath[:80] + '...' if len(expected_xpath) > 80 else expected_xpath
            act_short = actual_xpath[:80] + '...' if len(actual_xpath) > 80 else actual_xpath
            mismatches.append(f"xpath: 期望 '{exp_short}', 实际 '{act_short}' ✗")
    
    # 18. 验证 text_content（可能被截断）
    expected_text = (expected_info.get('text_content', '') or '').strip()
    if expected_text:
        total_count += 1
        actual_text = (actual_attrs.get('text', '') or '').strip()
        if truncated_match(expected_text, actual_text):
            matched_count += 1
            matches.append(f"text: ✓")
        else:
            # 截断长 text 避免刷屏
            exp_short = expected_text[:50] + '...' if len(expected_text) > 50 else expected_text
            act_short = (actual_text[:50] + '...') if actual_text and len(actual_text) > 50 else (actual_text or '无')
            mismatches.append(f"text: 期望 '{exp_short}', 实际 '{act_short}' ✗")
    
    # 注意：uid (data-webtasks-id) 不在这里验证，因为它是独立的指标
    
    # 判断是否匹配（必须完全匹配）
    if total_count == 0:
        return True, "no_attrs_to_check", 0, 0
    
    if matched_count == total_count:
        reason = "; ".join(matches)
        return True, reason, matched_count, total_count
    else:
        reason = "; ".join(mismatches)
        return False, reason, matched_count, total_count


# =============================================================================
# 数据加载函数（与 WebLINXStaticChecker 共用）
# =============================================================================

def load_replay(raw_data_path: str, demo_name: str, cache: Optional[Dict] = None) -> Optional[Dict]:
    """
    加载指定 demo 的 replay.json
    
    Args:
        raw_data_path: raw_data 根路径（包含 demonstrations/）
        demo_name: demo ID（如 'cptbbef'）
        cache: 可选的缓存字典，用于避免重复加载
        
    Returns:
        replay.json 的内容，或 None
    """
    import os
    import json
    
    # 使用缓存
    if cache is not None and demo_name in cache:
        return cache[demo_name]
    
    replay_path = os.path.join(raw_data_path, 'demonstrations', demo_name, 'replay.json')
    if not os.path.exists(replay_path):
        if cache is not None:
            cache[demo_name] = None
        return None
    
    try:
        with open(replay_path, 'r', encoding='utf-8') as f:
            replay = json.load(f)
        if cache is not None:
            cache[demo_name] = replay
        return replay
    except Exception:
        if cache is not None:
            cache[demo_name] = None
        return None


def get_page_path(raw_data_path: str, demo_name: str, turn_idx: int, 
                  replay_cache: Optional[Dict] = None) -> Optional[str]:
    """
    获取指定 turn 对应的 page 文件路径
    
    Args:
        raw_data_path: raw_data 根路径
        demo_name: demo ID
        turn_idx: turn 索引
        replay_cache: 可选的 replay.json 缓存
        
    Returns:
        page 文件的完整路径，或 None
    """
    import os
    
    replay = load_replay(raw_data_path, demo_name, replay_cache)
    if not replay:
        return None
    
    turns = replay.get('data', [])
    if turn_idx >= len(turns):
        return None
    
    turn = turns[turn_idx]
    state = turn.get('state', {})
    page = state.get('page')
    
    if not page:
        return None
    
    page_path = os.path.join(raw_data_path, 'demonstrations', demo_name, 'pages', page)
    if os.path.exists(page_path):
        return page_path
    return None


def get_scroll_info(raw_data_path: str, demo_name: str, turn_idx: int,
                    replay_cache: Optional[Dict] = None) -> Tuple[float, float]:
    """
    从 replay.json 获取滚动信息
    
    WebLINX 的 bbox 是视口坐标，需要通过 pageY - clientY 计算滚动偏移
    
    Args:
        raw_data_path: raw_data 根路径
        demo_name: demo ID
        turn_idx: turn 索引
        replay_cache: 可选的 replay.json 缓存
        
    Returns:
        (scroll_x, scroll_y) 滚动偏移，如果无法获取则返回 (0, 0)
    """
    replay = load_replay(raw_data_path, demo_name, replay_cache)
    if not replay:
        return 0.0, 0.0
    
    turns = replay.get('data', [])
    if turn_idx >= len(turns):
        return 0.0, 0.0
    
    turn = turns[turn_idx]
    action = turn.get('action', {})
    if not isinstance(action, dict):
        return 0.0, 0.0
    
    args = action.get('arguments', {})
    props = args.get('properties', {})
    
    page_x = props.get('pageX', 0)
    page_y = props.get('pageY', 0)
    client_x = props.get('clientX', 0)
    client_y = props.get('clientY', 0)
    
    scroll_x = page_x - client_x
    scroll_y = page_y - client_y
    
    return scroll_x, scroll_y


def read_html_file(path: str) -> str:
    """
    读取 HTML 文件，尝试多种编码
    
    Args:
        path: HTML 文件路径
        
    Returns:
        HTML 内容，读取失败返回空字符串
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    return ''


def load_raw_html_for_records(records: List, raw_data_path: str, show_progress: bool = True):
    """
    为 WebLINX records 加载 raw_html
    
    WebLINX 的 raw_html 存储在 pages/*.html 文件中，需要从 replay.json 获取文件名
    
    Args:
        records: Record 列表
        raw_data_path: raw_data 根路径（包含 demonstrations/）
        show_progress: 是否显示进度条
    
    Returns:
        (loaded_count, failed_count) 加载成功和失败的数量
    """
    # 可选的进度条
    if show_progress:
        try:
            from tqdm import tqdm
            records_iter = tqdm(records, desc="Loading raw_html")
        except ImportError:
            records_iter = records
    else:
        records_iter = records
    
    # 缓存 replay.json
    replay_cache = {}
    
    # 统计
    total_actions = sum(len(r.actions) for r in records)
    loaded_count = 0
    failed_count = 0
    
    # 遍历所有 records 和 actions
    for record in records_iter:
        # WebLINX 的真实 demo_id 存储在 metadata 中
        demo_name = record.metadata.get('demo_id') if record.metadata else None
        if not demo_name:
            # 所有 actions 都标记为失败
            failed_count += len(record.actions)
            continue
        
        for action in record.actions:
            # 从 metadata 获取 turn_idx
            turn_idx = action.metadata.get('turn', 0) if action.metadata else 0
            
            page_path = get_page_path(raw_data_path, demo_name, turn_idx, replay_cache)
            if page_path:
                raw_html = read_html_file(page_path)
                if raw_html:
                    action.raw_html = raw_html
                    loaded_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
    
    return loaded_count, failed_count


# =============================================================================
# 页面验证函数（Static 和 Dynamic Checker 共用）
# =============================================================================

def get_element_info(page, element) -> Dict:
    """
    获取元素的详细信息
    
    Args:
        page: Playwright page 对象
        element: 元素句柄
        
    Returns:
        元素信息字典
    """
    try:
        info = page.evaluate("""(el) => {
            if (!el || !el.tagName) return {};
            const rect = el.getBoundingClientRect();
            return {
                tag: el.tagName.toLowerCase(),
                uid: el.getAttribute('data-webtasks-id') || '',
                id: el.id || '',
                className: el.className || '',
                text: (el.textContent || '').replace(/\\s+/g, ' ').trim().substring(0, 200),
                type: el.type || '',
                placeholder: el.placeholder || '',
                value: el.value || '',
                name: el.name || '',
                role: el.getAttribute('role') || '',
                ariaLabel: el.getAttribute('aria-label') || '',
                href: el.getAttribute('href') || '',
                bbox: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
                visible: el.offsetParent !== null,
            };
        }""", element)
        return info or {}
    except:
        return {}


def verify_by_coords(
    page, 
    element_data: Dict,
    scroll_x: float = 0,
    scroll_y: float = 0,
    verbose: bool = True,
) -> Tuple[bool, str, Dict, Any]:
    """
    通过坐标定位验证元素（公共函数）
    
    WebLINX 的 bbox 是视口坐标，需要先滚动到正确位置后再用 elementsFromPoint 定位。
    
    定位策略：
    1. 滚动到数据收集时的位置
    2. 用 3 个检测点（左上、中心、右下）获取元素栈
    3. 遍历元素栈，验证属性是否匹配
    4. 如果顶层元素不匹配，搜索其子元素（有 5px 尺寸阈值过滤）
    
    注意：这是独立的指标，不涉及 UID 验证！
    
    Args:
        page: Playwright page 对象
        element_data: 元素信息（来自 parse_weblinx_candidate，包含 bbox、tag、class 等）
        scroll_x, scroll_y: 滚动偏移（从 replay.json 获取）
        verbose: 是否打印详细信息
        
    Returns:
        (success, reason, element_info, element_handle)
    """
    if not element_data:
        return False, "no_element_data", {}, None
    
    bbox = element_data.get('bbox')
    if not bbox:
        return False, "no_bbox", {}, None
    
    # 滚动到数据收集时的位置
    if scroll_y != 0 or scroll_x != 0:
        try:
            page.evaluate(f"window.scrollTo({scroll_x}, {scroll_y})")
            page.wait_for_timeout(300)
            if verbose:
                print(f"    [滚动] 已滚动到 scrollY={scroll_y:.0f}")
        except Exception as e:
            if verbose:
                print(f"    [滚动] 滚动失败: {e}")
    
    expected_tag = element_data.get('tag', '')
    expected_w = bbox.get('width', 0)
    expected_h = bbox.get('height', 0)
    expected_cx = bbox.get('x', 0) + expected_w / 2
    expected_cy = bbox.get('y', 0) + expected_h / 2
    
    # 定义 3 个检测点：左上、中心、右下
    check_points = [
        ('左上', bbox.get('x', 0), bbox.get('y', 0)),
        ('中心', expected_cx, expected_cy),
        ('右下', bbox.get('x', 0) + expected_w, bbox.get('y', 0) + expected_h),
    ]
    
    last_top_element = None
    
    try:
        # 遍历 3 个检测点
        for point_name, target_x, target_y in check_points:
            # 用 elementsFromPoint 获取该坐标下所有层叠元素
            all_elements = page.evaluate(f"""() => {{
                const elements = document.elementsFromPoint({target_x}, {target_y});
                return elements.map((el, idx) => ({{
                    index: idx,
                    tag: el.tagName.toLowerCase(),
                    id: el.id || '',
                    className: (el.className || '').toString().substring(0, 100),
                    rect: (() => {{
                        const r = el.getBoundingClientRect();
                        return {{x: r.x, y: r.y, width: r.width, height: r.height}};
                    }})()
                }}));
            }}""")
            
            if not all_elements:
                continue
            
            # 遍历所有层叠元素
            for elem_info in all_elements:
                elem_idx = elem_info['index']
                
                # 获取元素句柄
                element = page.evaluate_handle(f"""() => {{
                    const elements = document.elementsFromPoint({target_x}, {target_y});
                    return elements[{elem_idx}];
                }}""")
                
                # 检查 null 或 undefined
                is_invalid = page.evaluate("(el) => el === null || el === undefined || !el", element)
                if is_invalid:
                    continue
                
                last_top_element = element
                
                # 使用 verify_weblinx_element_match 验证属性是否匹配
                if element_data:
                    is_match, reason, matched, total = verify_weblinx_element_match(page, element, element_data)
                    if is_match:
                        if verbose:
                            print(f"    ✓ [{point_name}] 第{elem_idx}层元素直接匹配成功 ({matched}/{total})")
                            for attr_info in reason.split('; '):
                                if attr_info.strip():
                                    print(f"      {attr_info}")
                        try:
                            page.evaluate("(el) => el.style.border='3px solid green'", element)
                        except:
                            pass
                        element_info = get_element_info(page, element)
                        return True, f"success@{point_name}_layer{elem_idx} ({matched}/{total})", element_info, element
                
                # 搜索子元素（如果有期望的 tag）
                if expected_tag:
                    children_info = page.evaluate(f"""(el) => {{
                        const tag = '{expected_tag}';
                        const children = el.querySelectorAll(tag);
                        const results = [];
                        
                        for (let i = 0; i < children.length && i < 500; i++) {{
                            const child = children[i];
                            const rect = child.getBoundingClientRect();
                            
                            if (rect.width > 0 && rect.height > 0) {{
                                results.push({{
                                    index: i,
                                    rect: {{x: rect.x, y: rect.y, width: rect.width, height: rect.height}}
                                }});
                            }}
                        }}
                        return results;
                    }}""", element)
                    
                    if not children_info:
                        continue
                    
                    # 找最佳匹配的子元素（按大小和位置）
                    best_match_idx = -1
                    best_score = float('inf')
                    
                    for child in children_info:
                        rect = child['rect']
                        size_diff = abs(rect['width'] - expected_w) + abs(rect['height'] - expected_h)
                        child_cx = rect['x'] + rect['width'] / 2
                        child_cy = rect['y'] + rect['height'] / 2
                        pos_diff = ((child_cx - expected_cx)**2 + (child_cy - expected_cy)**2)**0.5
                        score = size_diff * 2 + pos_diff
                        
                        if score < best_score:
                            best_score = score
                            best_match_idx = child['index']
                    
                    if best_match_idx < 0:
                        continue
                    
                    best_child = page.evaluate_handle(f"""(el) => {{
                        const children = el.querySelectorAll('{expected_tag}');
                        return children[{best_match_idx}];
                    }}""", element)
                    
                    best_rect = page.evaluate("""(el) => {
                        const rect = el.getBoundingClientRect();
                        return {x: rect.x, y: rect.y, width: rect.width, height: rect.height};
                    }""", best_child)
                    
                    # 5px 尺寸阈值过滤
                    size_diff = abs(best_rect['width'] - expected_w) + abs(best_rect['height'] - expected_h)
                    
                    if size_diff > 5:
                        continue
                    
                    if element_data:
                        is_match, reason, matched, total = verify_weblinx_element_match(page, best_child, element_data)
                        if is_match:
                            if verbose:
                                print(f"    ✓ [{point_name}] 第{elem_idx}层的子元素匹配成功 ({matched}/{total})")
                                for attr_info in reason.split('; '):
                                    if attr_info.strip():
                                        print(f"      {attr_info}")
                            try:
                                page.evaluate("(el) => el.style.border='3px solid green'", best_child)
                            except:
                                pass
                            element_info = get_element_info(page, best_child)
                            return True, f"success_child@{point_name}_layer{elem_idx}[{best_match_idx}] ({matched}/{total})", element_info, best_child
        
        # 所有点都失败
        if last_top_element:
            is_match, reason, matched, total = verify_weblinx_element_match(page, last_top_element, element_data)
            top_info = page.evaluate("""(el) => ({
                tag: el.tagName.toLowerCase(),
                rect: el.getBoundingClientRect()
            })""", last_top_element)
            if verbose:
                print(f"    ✗ 3个检测点都未找到匹配元素")
                print(f"    [顶层元素] <{top_info['tag']}> @ ({top_info['rect']['x']:.0f},{top_info['rect']['y']:.0f})")
                if reason:
                    print(f"    ✗ 属性不匹配 ({matched}/{total}):")
                    for attr_info in reason.split('; '):
                        if attr_info.strip():
                            print(f"      {attr_info}")
            try:
                page.evaluate("(el) => el.style.border='3px solid orange'", last_top_element)
            except:
                pass
            element_info = get_element_info(page, last_top_element)
            return False, f"no_match ({matched}/{total})", element_info, last_top_element
        else:
            if verbose:
                print(f"    ✗ 未找到元素")
            return False, "element_not_found_at_coords", {}, None
            
    except Exception as e:
        if verbose:
            print(f"    ✗ 坐标定位错误: {e}")
        return False, f"coord_error: {str(e)}", {}, None


def verify_by_attrs(
    page,
    element_data: Dict,
    bbox: Dict = None,
    scroll_x: float = 0,
    scroll_y: float = 0,
    verbose: bool = True,
) -> Tuple[bool, str, Dict, Any]:
    """
    通过属性定位验证元素（公共函数）
    
    定位策略：
    1. 用 CSS 选择器找元素
    2. 逐个验证所有属性（包括 xpath 和 text）
    3. 找到第一个验证通过的就返回成功
    
    Args:
        page: Playwright page 对象
        element_data: 元素信息（来自 parse_weblinx_candidate）
        bbox: 目标元素的边界框（用于多匹配时选择最近的）
        scroll_x, scroll_y: 滚动偏移（从 replay.json 获取）
        verbose: 是否打印详细信息
        
    Returns:
        (success, reason, element_info, element_handle)
    """
    if not element_data:
        if verbose:
            print(f"    ❌ 无元素信息（数据集缺失）")
        return False, "no_element_data", {}, None
    
    # element_data 已经是解析后的格式（来自 parse_weblinx_candidate）
    tag_name = element_data.get('tag', '')
    
    # 获取 bbox（用于多匹配时选择）
    if bbox is None:
        bbox = element_data.get('bbox')
    
    # 如果有 bbox 用于坐标筛选，需要先滚动到正确位置
    if bbox and (scroll_y != 0 or scroll_x != 0):
        try:
            page.evaluate(f"window.scrollTo({scroll_x}, {scroll_y})")
            page.wait_for_timeout(300)
        except:
            pass
    
    # 用 CSS 选择器找元素
    selector, desc = build_css_selector(element_data)
    
    if not selector:
        if verbose:
            print(f"    ✗ 没有可用属性构建 CSS 选择器")
        return False, "no_css_selector", {'tag': tag_name}, None
    
    # 简化过长的选择器显示
    if verbose:
        if len(selector) > 80:
            print(f"    [CSS选择器] {desc}")
        else:
            print(f"    [CSS选择器] {selector}")
    
    try:
        elements = page.query_selector_all(selector)
    except Exception as e:
        if verbose:
            print(f"    ✗ CSS 选择器错误: {str(e)}")
        return False, f"css_error: {str(e)}", {'tag': tag_name}, None
    
    if not elements:
        if verbose:
            print(f"    ✗ 未找到元素")
        return False, "css_not_found", {'tag': tag_name}, None
    
    # 如果多个元素，按坐标距离排序
    if len(elements) > 1 and bbox:
        target_x = bbox['x'] + bbox['width'] / 2
        target_y = bbox['y'] + bbox['height'] / 2
        
        def get_distance(e):
            try:
                rect = e.bounding_box()
                if rect:
                    elem_cx = rect['x'] + rect['width'] / 2
                    elem_cy = rect['y'] + rect['height'] / 2
                    return ((elem_cx - target_x) ** 2 + (elem_cy - target_y) ** 2) ** 0.5
            except:
                pass
            return float('inf')
        
        elements = sorted(elements, key=get_distance)
    
    # 多元素时提示
    if verbose and len(elements) > 1:
        print(f"    找到 {len(elements)} 个候选，按坐标距离排序验证")
    
    # 逐个验证，找到第一个通过验证的
    fail_reasons = []  # 收集失败原因
    for i, element in enumerate(elements):
        is_match, reason, matched, total = verify_weblinx_element_match(page, element, element_data)
        
        if is_match:
            element_info = get_element_info(page, element)
            if verbose:
                if len(elements) == 1:
                    print(f"    ✓ 匹配成功 ({matched}/{total})")
                else:
                    print(f"    ✓ 第{i+1}个元素匹配成功 ({matched}/{total})")
                # 打印匹配的属性详情
                for attr_info in reason.split('; '):
                    if attr_info.strip():
                        print(f"      {attr_info}")
            return True, f"match ({matched}/{total})", element_info, element
        else:
            # 记录失败原因
            fail_reasons.append((i + 1, reason, matched, total))
    
    # 所有元素都验证失败
    element_info = get_element_info(page, elements[0]) if elements else {}
    if verbose:
        print(f"    ✗ 所有 {len(elements)} 个候选元素验证失败")
        # 打印每个元素的失败原因（最多显示前 3 个）
        for idx, reason, matched, total in fail_reasons[:3]:
            print(f"      - 元素{idx}: ({matched}/{total})")
            for attr_info in reason.split('; '):
                if attr_info.strip():
                    print(f"        {attr_info}")
        if len(fail_reasons) > 3:
            print(f"      ... 还有 {len(fail_reasons) - 3} 个元素失败")
    return False, "all_verify_failed", element_info, None


