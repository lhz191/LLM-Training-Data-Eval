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


def find_candidate_by_uid(candidates: list, target_uid: str) -> Optional[dict]:
    """根据 uid 在 candidates 中找到目标元素"""
    if not candidates or not target_uid:
        return None
    for cand in candidates:
        if cand.get('uid') == target_uid:
            return cand
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
                mismatches.append(f"{attr_name}: 期望 '{expected_val}', 实际 '{actual_val or '无'}' ✗")
    
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
            mismatches.append(f"xpath: 期望 '{expected_xpath}', 实际 '{actual_xpath}' ✗")
    
    # 18. 验证 text_content（可能被截断）
    expected_text = (expected_info.get('text_content', '') or '').strip()
    if expected_text:
        total_count += 1
        actual_text = (actual_attrs.get('text', '') or '').strip()
        if truncated_match(expected_text, actual_text):
            matched_count += 1
            matches.append(f"text: ✓")
        else:
            mismatches.append(f"text: 期望 '{expected_text}', 实际 '{actual_text or '无'}' ✗")
    
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
