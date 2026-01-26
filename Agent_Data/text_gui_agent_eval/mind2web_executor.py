#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mind2Web 静态可执行性检查器

基于 MHTML 快照验证 Record 中每个 Action 是否可定位到目标元素。

验证方式（两种指标并行）：
1. 坐标定位 (verify_by_coords1): 使用 bounding_box 坐标定位元素
2. 属性定位 (verify_by_attrs): 使用 tag/class/id 等属性定位元素

依赖：
- raw_dump 数据集（包含 MHTML 快照）
- Playwright 浏览器

Usage:
    from mind2web_executor import Mind2WebStaticChecker
    from loaders import Mind2WebLoader
    
    loader = Mind2WebLoader('/path/to/mind2web/data')
    checker = Mind2WebStaticChecker(
        raw_dump_path='/path/to/raw_dump',
        headless=True
    )
    
    for record in loader.iterate():
        errors, warnings, stats = checker.check(record)
        print(f"坐标定位: {stats['coord_rate']:.1%}")
        print(f"属性定位: {stats['attr_rate']:.1%}")
"""
import os
import time
from typing import List, Dict, Any, Tuple, Optional

from text_gui_executor import StaticExecutabilityChecker, register_static_checker
from data_types import Record, Action

try:
    from playwright.sync_api import sync_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    print("⚠️ Playwright not installed. Install with: pip install playwright && playwright install")

# =============================================================================
# 常量（来自 verify_dynamic.py）
# =============================================================================

# 官方 Mind2Web 视口大小
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 1080


# =============================================================================
# 内化的辅助函数（来自 verify_dynamic.py）
# =============================================================================

def is_dynamic_class(c):
    """判断是否是动态生成的 class（CSS-in-JS 等）或无效的 CSS 类名"""
    if not c:
        return True
    return (
        c.startswith('css-') or      # Emotion/styled-components
        c.startswith('jss') or       # JSS
        (len(c) > 0 and c[0].isdigit()) or  # 数字开头
        (len(c) <= 10 and any(ch.isdigit() for ch in c)) or  # 短且含数字
        ':' in c or                  # Tailwind 变体类 (hover:xxx, focus:xxx)
        '[' in c or ']' in c         # Tailwind 任意值类 ([color:red])
    )


def escape_css_value(s):
    """转义 CSS 属性值中的特殊字符"""
    if not s:
        return s
    # 转义常见特殊字符
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace("'", "\\'")
    s = s.replace('\n', ' ')
    s = s.replace('\r', ' ')
    return s


def parse_candidate(candidate):
    """
    解析 pos_candidate，提取所有属性
    """
    import json
    
    result = {
        'tag': candidate.get('tag', ''),
        'bbox': None,
        
        # CSS 可筛选的属性
        'class': None,
        'id': None,
        'name': None,
        'type': None,
        'aria_label': None,
        'aria_description': None,
        'role': None,
        'placeholder': None,
        'title': None,
        'alt': None,
        'value': None,
        'label': None,
        
        # 文本/状态匹配的属性
        'input_value': None,
        'input_checked': None,
        'text_value': None,
        'is_clickable': None,
    }
    
    attrs_str = candidate.get('attributes', '{}')
    try:
        attrs = json.loads(attrs_str)
        
        # 提取坐标
        bbox_str = attrs.get('bounding_box_rect', '')
        if bbox_str:
            parts = bbox_str.split(',')
            if len(parts) == 4:
                x, y, w, h = map(float, parts)
                result['bbox'] = {'x': x, 'y': y, 'width': w, 'height': h}
        
        # 提取属性
        result['class'] = attrs.get('class', '')
        result['id'] = attrs.get('id', '')
        result['name'] = attrs.get('name', '')
        result['type'] = attrs.get('type', '')
        result['aria_label'] = attrs.get('aria_label', '') or attrs.get('aria-label', '')
        result['aria_description'] = attrs.get('aria_description', '') or attrs.get('aria-description', '')
        result['role'] = attrs.get('role', '')
        result['placeholder'] = attrs.get('placeholder', '')
        result['title'] = attrs.get('title', '')
        result['alt'] = attrs.get('alt', '')
        result['value'] = attrs.get('value', '')
        result['label'] = attrs.get('label', '')
        
        result['input_value'] = attrs.get('input_value', '')
        result['input_checked'] = attrs.get('input_checked')
        result['text_value'] = attrs.get('text_value', '')
        result['is_clickable'] = attrs.get('is_clickable')
        
    except:
        pass
    
    return result


def find_element_by_all_attributes(page, info, bbox=None):
    """
    用所有可用属性组合定位元素
    
    属性分类：
    ===== CSS 可筛选 (有值就用，不做任何过滤) =====
    class (89.3%), type (23.1%), id (21.0%), aria_label (17.6%), role (15.5%),
    name (10.1%), value (9.9%), placeholder (7.7%), title (4.3%), alt (2.0%),
    label (0.0%), aria_description (0.0%), data_pw_testid_buckeye_candidate (100%)
    
    ===== 文本/状态匹配 (CSS不支持，后续过滤) =====
    is_clickable (42.8%), input_value (12.6%), input_checked (0.2%), text_value (0.1%)
    
    Args:
        page: Playwright page 对象
        info: 元素属性信息
        bbox: 坐标信息 {'x': x, 'y': y, 'width': w, 'height': h}，用于多匹配时选择最近的
    
    Returns:
        (element, method_description) 或 (None, None)
    """
    # ===== 提取属性 =====
    tag = info.get('tag', '')
    
    # CSS 可筛选的属性
    cls = info.get('class', '')
    elem_type = info.get('type', '')
    elem_id = info.get('id', '')
    aria_label = info.get('aria_label', '')
    role = info.get('role', '')
    name = info.get('name', '')
    value = info.get('value', '')
    placeholder = info.get('placeholder', '')
    title = info.get('title', '')
    alt = info.get('alt', '')
    label = info.get('label', '')
    aria_description = info.get('aria_description', '')
    data_pw_testid = info.get('data_pw_testid_buckeye_candidate', '')
    
    # ===== 文本/状态匹配的属性 (CSS不支持，需后续过滤/验证) =====
    text_value = info.get('text_value', '')        # 0.1% - 文本内容匹配
    input_value = info.get('input_value', '')      # 12.6% - 输入框当前值
    input_checked = info.get('input_checked')      # 0.2% - 复选框/单选框状态
    is_clickable = info.get('is_clickable')        # 42.8% - 可点击状态
    
    # ===== 构建 CSS 选择器 =====
    selector_parts = []
    conditions_desc = []
    
    # 有值就加，不做任何过滤
    if tag:
        selector_parts.append(tag)
        conditions_desc.append(f"tag={tag}")
    
    if elem_id:
        selector_parts.append(f'[id="{escape_css_value(elem_id)}"]')
        conditions_desc.append(f"id")
    
    if name:
        selector_parts.append(f'[name="{escape_css_value(name)}"]')
        conditions_desc.append(f"name")
    
    if elem_type:
        selector_parts.append(f'[type="{escape_css_value(elem_type)}"]')
        conditions_desc.append(f"type")
    
    if role:
        selector_parts.append(f'[role="{escape_css_value(role)}"]')
        conditions_desc.append(f"role")
    
    if aria_label:
        selector_parts.append(f'[aria-label="{escape_css_value(aria_label)}"]')
        conditions_desc.append(f"aria-label")
    
    if aria_description:
        selector_parts.append(f'[aria-description="{escape_css_value(aria_description)}"]')
        conditions_desc.append(f"aria-description")
    
    if placeholder:
        selector_parts.append(f'[placeholder="{escape_css_value(placeholder)}"]')
        conditions_desc.append(f"placeholder")
    
    if title:
        selector_parts.append(f'[title="{escape_css_value(title)}"]')
        conditions_desc.append(f"title")
    
    if alt:
        selector_parts.append(f'[alt="{escape_css_value(alt)}"]')
        conditions_desc.append(f"alt")
    
    # value 是数据收集时输入框的值，真实网站上可能不同或为空，不用于定位
    if value:
        selector_parts.append(f'[value="{escape_css_value(value)}"]')
        conditions_desc.append(f"value")
    
    if label:
        selector_parts.append(f'[label="{escape_css_value(label)}"]')
        conditions_desc.append(f"label")
    
    # data-pw-testid-buckeye-candidate 是 Playwright 数据收集时添加的标记，真实网站上不存在，不用于定位
    # if data_pw_testid:
    #     selector_parts.append(f'[data-pw-testid-buckeye-candidate="{escape_css_value(data_pw_testid)}"]')
    #     conditions_desc.append(f"pw_testid")
    
    if cls:
        # 过滤掉动态生成的 class 名（CSS-in-JS 生成的，每次构建都会变）
        # 动态 class 特征：css-xxx, jss-xxx, jss数字, 纯数字开头
        stable_classes = []
        dynamic_classes = []
        for c in cls.split():
            if not c:
                continue
            # 判断是否是动态 class（使用统一的函数）
            if is_dynamic_class(c):
                dynamic_classes.append(c)
            else:
                stable_classes.append(c)
        
        # 只用稳定的 class
        for c in stable_classes:
            selector_parts.append(f'.{c}')
        
        if stable_classes:
            conditions_desc.append(f"class({len(stable_classes)}个)")
        if dynamic_classes:
            print(f"    [跳过动态class] {', '.join(dynamic_classes[:3])}{'...' if len(dynamic_classes) > 3 else ''}")
    
    # ===== 检查条件 =====
    # 即使只有 tag，也尝试定位（多匹配时用坐标选最近的）
    if not selector_parts:
        print(f"    ⚠ 没有可用属性")
        return None, None
    
    if len(selector_parts) == 1 and tag:
        print(f"    ⚠ 只有 tag，可能匹配多个元素")
    
    # ===== 执行 CSS 选择器查找 =====
    selector = ''.join(selector_parts)
    desc = '+'.join(conditions_desc)
    
    print(f"    [CSS选择器] {selector}")
    
    try:
        elements = page.query_selector_all(selector)
        
        # ===== 文本/状态过滤 =====
        
        # text_value 过滤（文本内容）
        if text_value and elements:
            print(f"    [文本过滤] 需要包含: '{text_value}'")
            original_count = len(elements)
            filtered = []
            for e in elements:
                try:
                    elem_text = e.text_content() or ''
                    if text_value in elem_text:
                        filtered.append(e)
                except:
                    pass
            elements = filtered
            if original_count != len(elements):
                print(f"    [文本过滤] {original_count} -> {len(elements)} 个")
            desc += "+text"
        
        # input_value 过滤（输入框当前值）
        if input_value and elements:
            print(f"    [input_value过滤] 需要值为: '{input_value}'")
            original_count = len(elements)
            filtered = []
            for e in elements:
                try:
                    # 用 DOM 属性匹配（兼容静态和动态场景）
                    val = page.evaluate("(el) => el.value", e) or ''
                    if val == input_value:
                        filtered.append(e)
                except:
                    pass
            elements = filtered
            if original_count != len(elements):
                print(f"    [input_value过滤] {original_count} -> {len(elements)} 个")
            desc += "+input_value"
        
        # input_checked 过滤（复选框/单选框状态）
        if input_checked is not None and elements:
            # 转换为布尔值统一比较
            expected_checked_bool = str(input_checked).lower() == 'true'
            print(f"    [input_checked过滤] 需要选中状态: {expected_checked_bool}")
            original_count = len(elements)
            filtered = []
            for e in elements:
                try:
                    checked = e.is_checked()
                    if checked == expected_checked_bool:
                        filtered.append(e)
                except:
                    pass
            elements = filtered
            if original_count != len(elements):
                print(f"    [input_checked过滤] {original_count} -> {len(elements)} 个")
            desc += "+checked"
        
        # is_clickable 过滤（可点击状态）
        # 使用与 verify_element_match 一致的判断逻辑
        if is_clickable is not None and elements:
            # 转换为布尔值统一比较
            expected_clickable_bool = str(is_clickable).lower() == 'true'
            print(f"    [is_clickable过滤] 需要可点击: {expected_clickable_bool}")
            original_count = len(elements)
            filtered = []
            for e in elements:
                try:
                    # 使用与 verify_element_match 一致的 JavaScript 判断
                    # 而不是 Playwright 的 is_visible()，因为后者在 MHTML 中可能过于严格
                    clickable = page.evaluate("""(el) => {
                        const is_displayed = el.offsetParent !== null;
                        const is_enabled = !el.disabled;
                        return is_displayed && is_enabled;
                    }""", e)
                    if clickable == expected_clickable_bool:
                        filtered.append(e)
                except:
                    pass
            elements = filtered
            if original_count != len(elements):
                print(f"    [is_clickable过滤] {original_count} -> {len(elements)} 个")
            desc += "+clickable"
        
        # ===== 返回结果 =====
        if len(elements) == 1:
            print(f"    ✓ 精确匹配: {desc}")
            return elements[0], f"精确匹配({desc})"
        elif len(elements) > 1:
            # 多个匹配时，用坐标选最近的
            if bbox:
                target_x = bbox['x'] + bbox['width'] / 2
                target_y = bbox['y'] + bbox['height'] / 2
                print(f"    ⚠ 找到 {len(elements)} 个，用坐标 ({target_x:.0f}, {target_y:.0f}) 选最近的")
                
                # 获取当前滚动位置，用于将视口坐标转换为文档坐标
                scroll_top = page.evaluate("window.pageYOffset")
                
                best_element = None
                best_distance = float('inf')
                
                for elem in elements:
                    try:
                        rect = elem.bounding_box()  # 返回的是视口坐标！
                        if rect:
                            elem_x = rect['x'] + rect['width'] / 2
                            # 将视口 y 坐标转换为文档 y 坐标
                            elem_y = rect['y'] + scroll_top + rect['height'] / 2
                            distance = ((elem_x - target_x) ** 2 + (elem_y - target_y) ** 2) ** 0.5
                            if distance < best_distance:
                                best_distance = distance
                                best_element = elem
                    except:
                        continue
                
                if best_element:
                    print(f"    ✓ 选择距离最近的元素 (距离: {best_distance:.0f}px)")
                    return best_element, f"多匹配({len(elements)}个,坐标选择)"
                else:
                    print(f"    ⚠ 无法获取元素坐标，取第一个")
                    return elements[0], f"多匹配({len(elements)}个)"
            else:
                print(f"    ⚠ 找到 {len(elements)} 个，无坐标信息，取第一个")
                return elements[0], f"多匹配({len(elements)}个)"
        else:
            print(f"    ✗ 未找到元素")
            return None, None
            
    except Exception as e:
        print(f"    ✗ 选择器错误: {e}")
    return None, None


def verify_element_match(page, element_handle, expected_info):
    """
    验证找到的元素是否与数据集描述匹配
    
    - 过滤动态 class (css-xxx, jss-xxx 等)
    - 有值就验证，不做任何特殊过滤
    
    Returns:
        (is_match, mismatch_reason, matched_attrs, total_attrs)
    """
    if not element_handle:
        return False, "element_is_none", 0, 0
    
    # 获取实际元素属性（通过 JavaScript）
    actual_attrs = page.evaluate("""(element) => {
        return {
            tag: element.tagName.toLowerCase(),
            class: element.getAttribute('class') || '',
            id: element.getAttribute('id') || '',
            name: element.getAttribute('name') || '',
            placeholder: element.getAttribute('placeholder') || '',
            title: element.getAttribute('title') || '',
            aria_label: element.getAttribute('aria-label') || '',
            aria_description: element.getAttribute('aria-description') || '',
            role: element.getAttribute('role') || '',
            type: element.getAttribute('type') || '',
            alt: element.getAttribute('alt') || '',
            value: element.getAttribute('value') || '',
            label: element.getAttribute('label') || '',
            text: element.textContent || '',
            is_displayed: element.offsetParent !== null,
            is_enabled: !element.disabled,
            input_value: element.value || '',
            input_checked: element.checked
        };
    }""", element_handle)
    
    actual_tag = actual_attrs['tag']
    actual_class = actual_attrs['class']
    actual_id = actual_attrs['id']
    actual_name = actual_attrs['name']
    actual_placeholder = actual_attrs['placeholder']
    actual_title = actual_attrs['title']
    actual_aria_label = actual_attrs['aria_label']
    actual_aria_description = actual_attrs['aria_description']
    actual_role = actual_attrs['role']
    actual_type = actual_attrs['type']
    actual_alt = actual_attrs['alt']
    actual_label = actual_attrs['label']
    actual_text = actual_attrs['text']
    actual_is_clickable = actual_attrs['is_displayed'] and actual_attrs['is_enabled']
    actual_value = actual_attrs['value']
    actual_input_value = actual_attrs['input_value']
    actual_input_checked = actual_attrs['input_checked']
    
    # 期望的属性
    expected_tag = (expected_info.get('tag') or '').lower()
    expected_class = expected_info.get('class') or ''
    expected_id = expected_info.get('id') or ''
    expected_name = expected_info.get('name') or ''
    expected_placeholder = expected_info.get('placeholder') or ''
    expected_title = expected_info.get('title') or ''
    expected_aria_label = expected_info.get('aria_label') or ''
    expected_aria_description = expected_info.get('aria_description') or ''
    expected_role = expected_info.get('role') or ''
    expected_type = expected_info.get('type') or ''
    expected_alt = expected_info.get('alt') or ''
    expected_label = expected_info.get('label') or ''
    expected_text_value = expected_info.get('text_value') or ''
    expected_is_clickable = expected_info.get('is_clickable')
    expected_value = expected_info.get('value') or ''
    expected_input_value = expected_info.get('input_value') or ''
    expected_input_checked = expected_info.get('input_checked')
    
    mismatches = []
    matches = []  # 记录匹配的属性
    matched_count = 0
    total_count = 0
    
    # 验证 tag
    if expected_tag:
        total_count += 1
        if actual_tag == expected_tag:
            matched_count += 1
            matches.append(f"tag: {actual_tag} ✓")
        else:
            mismatches.append(f"tag: 期望 {expected_tag}, 实际 {actual_tag} ✗")
    
    # 验证 id
    if expected_id:
        total_count += 1
        if actual_id == expected_id:
            matched_count += 1
            matches.append(f"id: {actual_id} ✓")
        else:
            mismatches.append(f"id: 期望 {expected_id}, 实际 {actual_id or '无'} ✗")
    
    # 验证 name
    if expected_name:
        total_count += 1
        if actual_name == expected_name:
            matched_count += 1
            matches.append(f"name: {actual_name} ✓")
        else:
            mismatches.append(f"name: 期望 {expected_name}, 实际 {actual_name or '无'} ✗")
    
    # 验证 class（过滤动态 class）
    if expected_class:
        expected_classes = set(expected_class.split())
        actual_classes = set(actual_class.split())
        
        # 过滤掉动态 class
        stable_expected_classes = {c for c in expected_classes if not is_dynamic_class(c)}
        
        if stable_expected_classes:
            total_count += 1
            missing_classes = stable_expected_classes - actual_classes
            matched_classes = stable_expected_classes & actual_classes
            if not missing_classes:
                matched_count += 1
                matches.append(f"class: 全部匹配 ({len(stable_expected_classes)}个) ✓")
            else:
                # 显示期望和实际的 class
                expected_str = ' '.join(sorted(stable_expected_classes))
                actual_str = actual_class if actual_class else '无'
                mismatches.append(f"class: 期望 [{expected_str}], 实际 [{actual_str}] ✗")
    
    # 验证 type
    if expected_type:
        total_count += 1
        if actual_type == expected_type:
            matched_count += 1
            matches.append(f"type: {actual_type} ✓")
        else:
            mismatches.append(f"type: 期望 {expected_type}, 实际 {actual_type or '无'} ✗")
    
    # 验证 role
    if expected_role:
        total_count += 1
        if actual_role == expected_role:
            matched_count += 1
            matches.append(f"role: {actual_role} ✓")
        else:
            mismatches.append(f"role: 期望 {expected_role}, 实际 {actual_role or '无'} ✗")
    
    # 验证 aria-label
    if expected_aria_label:
        total_count += 1
        if expected_aria_label == actual_aria_label or expected_aria_label in actual_aria_label:
            matched_count += 1
            matches.append(f"aria-label: {actual_aria_label} ✓")
        else:
            mismatches.append(f"aria-label: 期望 '{expected_aria_label}', 实际 '{actual_aria_label or '无'}' ✗")
    
    # 验证 aria-description
    if expected_aria_description:
        total_count += 1
        if expected_aria_description == actual_aria_description or expected_aria_description in actual_aria_description:
            matched_count += 1
            matches.append(f"aria-description: {actual_aria_description} ✓")
        else:
            mismatches.append(f"aria-description: 期望 '{expected_aria_description}', 实际 '{actual_aria_description or '无'}' ✗")
    
    # 验证 placeholder
    if expected_placeholder:
        total_count += 1
        if expected_placeholder == actual_placeholder or expected_placeholder in actual_placeholder:
            matched_count += 1
            matches.append(f"placeholder: {actual_placeholder} ✓")
        else:
            mismatches.append(f"placeholder: 期望 '{expected_placeholder}', 实际 '{actual_placeholder or '无'}' ✗")
    
    # 验证 title
    if expected_title:
        total_count += 1
        if expected_title == actual_title or expected_title in actual_title:
            matched_count += 1
            matches.append(f"title: {actual_title} ✓")
        else:
            mismatches.append(f"title: 期望 '{expected_title}', 实际 '{actual_title or '无'}' ✗")
    
    # 验证 alt
    if expected_alt:
        total_count += 1
        if expected_alt == actual_alt or expected_alt in actual_alt:
            matched_count += 1
            matches.append(f"alt: {actual_alt} ✓")
        else:
            mismatches.append(f"alt: 期望 '{expected_alt}', 实际 '{actual_alt or '无'}' ✗")
    
    # 验证 label
    if expected_label:
        total_count += 1
        if expected_label == actual_label or expected_label in actual_label:
            matched_count += 1
            matches.append(f"label: {actual_label} ✓")
        else:
            mismatches.append(f"label: 期望 '{expected_label}', 实际 '{actual_label or '无'}' ✗")
    
    # 验证 text_value（文本内容）
    if expected_text_value:
        total_count += 1
        if expected_text_value == actual_text or expected_text_value in actual_text:
            matched_count += 1
            matches.append(f"text: 包含 '{expected_text_value}' ✓")
        else:
            mismatches.append(f"text: 期望 '{expected_text_value}', 实际 '{actual_text[:50] if actual_text else '无'}' ✗")
    
    # 验证 value（HTML 属性值）
    if expected_value:
        total_count += 1
        if actual_value == expected_value:
            matched_count += 1
            matches.append(f"value: {actual_value} ✓")
        else:
            mismatches.append(f"value: 期望 '{expected_value}', 实际 '{actual_value or '无'}' ✗")
    
    # 验证 input_value（输入框当前值）
    if expected_input_value:
        total_count += 1
        if actual_input_value == expected_input_value:
            matched_count += 1
            matches.append(f"input_value: {actual_input_value} ✓")
        else:
            mismatches.append(f"input_value: 期望 '{expected_input_value}', 实际 '{actual_input_value or '无'}' ✗")
    
    # 验证 input_checked（复选框/单选框选中状态）
    if expected_input_checked is not None:
        total_count += 1
        expected_checked_bool = str(expected_input_checked).lower() == 'true'
        actual_checked_bool = actual_input_checked == True
        if expected_checked_bool == actual_checked_bool:
            matched_count += 1
            matches.append(f"input_checked: {actual_checked_bool} ✓")
        else:
            mismatches.append(f"input_checked: 期望 {expected_checked_bool}, 实际 {actual_checked_bool} ✗")
    
    # 验证 is_clickable
    if expected_is_clickable is not None:
        total_count += 1
        expected_clickable_bool = str(expected_is_clickable).lower() == 'true'
        if expected_clickable_bool == actual_is_clickable:
            matched_count += 1
            matches.append(f"is_clickable: {actual_is_clickable} ✓")
        else:
            mismatches.append(f"is_clickable: 期望 {expected_clickable_bool}, 实际 {actual_is_clickable} ✗")
    
    # 判断结果
    if total_count == 0:
        return True, "no_attrs_to_verify", 0, 0
    
    # 组合匹配和不匹配的信息
    all_results = matches + mismatches
    result_str = "; ".join(all_results)
    
    if mismatches:
        return False, result_str, matched_count, total_count
    
    return True, result_str, matched_count, total_count


# =============================================================================
# Mind2Web 静态可执行性检查器
# =============================================================================

class Mind2WebStaticChecker(StaticExecutabilityChecker):
    """
    Mind2Web 静态可执行性检查器
    
    使用 MHTML 快照验证 Action 是否可定位到目标元素。
    同时报告坐标定位和属性定位两种方式的成功率。
    
    Args:
        raw_dump_path: raw_dump 数据集路径
        headless: 是否使用无头浏览器模式
        timeout: 页面加载超时时间（秒）
    """
    
    # 默认 raw_dump 路径
    DEFAULT_RAW_DUMP_PATH = '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/Mind2Web/raw_dump'
    
    def __init__(
        self,
        raw_dump_path: Optional[str] = None,
        headless: bool = False,
        timeout: int = 30,
    ):
        if not HAS_PLAYWRIGHT:
            raise ImportError("Playwright is required. Install with: pip install playwright && playwright install")
        
        self.raw_dump_path = raw_dump_path or self.DEFAULT_RAW_DUMP_PATH
        self.headless = headless
        self.timeout = timeout * 1000  # 转换为毫秒
        
        # 浏览器实例（延迟初始化）
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
    
    def _ensure_browser(self):
        """确保浏览器已启动"""
        if self._page is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            context = self._browser.new_context(
                viewport={'width': VIEWPORT_WIDTH, 'height': VIEWPORT_HEIGHT}
            )
            self._page = context.new_page()
    
    def _close_browser(self):
        """关闭浏览器"""
        if self._browser:
            self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
    
    def _find_mhtml_path(self, annotation_id: str, action_uid: str) -> Optional[str]:
        """
        根据 annotation_id 和 action_uid 查找 MHTML 文件路径
        
        Args:
            annotation_id: Record 的 annotation_id
            action_uid: Action 的 action_uid
            
        Returns:
            MHTML 文件路径，如果不存在返回 None
        """
        mhtml_dir = os.path.join(
            self.raw_dump_path, 'task', annotation_id, 'processed', 'snapshots'
        )
        before_path = os.path.join(mhtml_dir, f'{action_uid}_before.mhtml')
        
        if os.path.exists(before_path):
            return before_path
        return None
    
    # =========================================================================
    # 内化的验证函数（来自 verify_dynamic.py，改为直接使用 Action 对象）
    # =========================================================================
    
    def _verify_by_coords(self, action: Action) -> Tuple[bool, str, Any]:
        """
        坐标定位验证（内化自 verify_by_coords1）
        
        直接使用 Action.target_element，无需构建 raw_action dict。
        
        Args:
            action: Action 对象
            
        Returns:
            (success, reason, element)
        """
        target_element = action.target_element
        
        if not target_element:
            return False, "no_target_element", None
        
        # 解析候选信息
        candidate_info = parse_candidate(target_element)
        bbox = candidate_info.get('bbox')
        expected_tag = candidate_info.get('tag', '').lower()
        
        if not bbox:
            return False, "no_bbox", None
        
        page = self._page
        
        print(f"    [坐标] 数据集 bbox: ({bbox['x']:.1f}, {bbox['y']:.1f}, {bbox['width']:.1f}x{bbox['height']:.1f})")
        print(f"    [期望] tag={expected_tag}, size={bbox['width']:.0f}x{bbox['height']:.0f}")
        
        # 获取页面高度
        try:
            page_height = page.evaluate("document.documentElement.scrollHeight")
        except Exception as e:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=10000)
                page_height = page.evaluate("document.documentElement.scrollHeight")
            except Exception as e2:
                return False, f"navigation_error: {e2}", None
        
        # 滚动到目标位置
        center_y = bbox['y'] + bbox['height'] / 2
        scroll_y = max(0, center_y - 300)
        page.evaluate(f"window.scrollTo(0, {scroll_y})")
        time.sleep(1)
        
        scroll_top = page.evaluate("window.pageYOffset")
        
        # 定义 3 个检测点：左上、中心、右下
        check_points = [
            ('左上', bbox['x'], bbox['y']),
            ('中心', bbox['x'] + bbox['width'] / 2, bbox['y'] + bbox['height'] / 2),
            ('右下', bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
        ]
        
        expected_w = bbox['width']
        expected_h = bbox['height']
        expected_cx = bbox['x'] + expected_w / 2
        expected_cy = bbox['y'] + expected_h / 2
        
        last_top_element = None
        
        # 遍历 3 个检测点
        for point_name, target_x, target_y in check_points:
            viewport_y = target_y - scroll_top
            
            if viewport_y < 0 or viewport_y > VIEWPORT_HEIGHT:
                continue
            
            # 用 elementsFromPoint 获取该坐标下所有层叠元素
            all_elements = page.evaluate(f"""() => {{
                const elements = document.elementsFromPoint({target_x}, {viewport_y});
                return elements.map((el, idx) => ({{
                    index: idx,
                    tag: el.tagName.toLowerCase(),
                    id: el.id || '',
                    className: (el.className || '').toString().substring(0, 50),
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
                
                element = page.evaluate_handle(f"""() => {{
                    const elements = document.elementsFromPoint({target_x}, {viewport_y});
                    return elements[{elem_idx}];
                }}""")
                
                is_null = page.evaluate("(el) => el === null", element)
                if is_null:
                    continue
                
                last_top_element = element
                
                # 检查该元素本身是否匹配
                is_match, reason, matched, total = verify_element_match(page, element, candidate_info)
                if is_match:
                    print(f"    [{point_name}] 第{elem_idx}层元素直接匹配成功")
                    print(f"    ✓ 匹配成功 ({matched}/{total}): {reason}")
                    try:
                        page.evaluate("(el) => el.style.border='3px solid green'", element)
                    except:
                        pass
                    return True, f"success@{point_name}_layer{elem_idx} ({matched}/{total})", element
                
                # 搜索子元素
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
                
                # 找最佳匹配的子元素
                best_match_idx = -1
                best_score = float('inf')
                
                for child in children_info:
                    rect = child['rect']
                    size_diff = abs(rect['width'] - expected_w) + abs(rect['height'] - expected_h)
                    child_cx = rect['x'] + rect['width'] / 2
                    child_cy = rect['y'] + scroll_top + rect['height'] / 2
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
                
                size_diff = abs(best_rect['width'] - expected_w) + abs(best_rect['height'] - expected_h)
                
                if size_diff > 5:
                    continue
                
                is_match, reason, matched, total = verify_element_match(page, best_child, candidate_info)
                if is_match:
                    print(f"    [{point_name}] 第{elem_idx}层的子元素匹配")
                    print(f"    ✓ 子元素匹配成功 ({matched}/{total}): {reason}")
                    try:
                        page.evaluate("(el) => el.style.border='3px solid green'", best_child)
                    except:
                        pass
                    return True, f"success_child@{point_name}_layer{elem_idx}[{best_match_idx}] ({matched}/{total})", best_child
        
        # 所有点都失败
        if last_top_element:
            top_info = page.evaluate("""(el) => ({
                tag: el.tagName.toLowerCase(),
                rect: el.getBoundingClientRect()
            })""", last_top_element)
            print(f"    [顶层元素] <{top_info['tag']}> @ ({top_info['rect']['x']:.0f},{top_info['rect']['y']:.0f})")
            print(f"    ✗ 3个检测点都未找到匹配的 <{expected_tag}> 元素")
            try:
                page.evaluate("(el) => el.style.border='3px solid orange'", last_top_element)
            except:
                pass
            return False, "no_match_all_points", last_top_element
        else:
            return False, "element_not_found_at_coords", None
    
    def _verify_by_attrs(self, action: Action) -> Tuple[bool, str, Any]:
        """
        属性定位验证（内化自 verify_by_attrs）
        
        直接使用 Action.target_element。
        
        Args:
            action: Action 对象
            
        Returns:
            (success, reason, element)
        """
        target_element = action.target_element
        
        if not target_element:
            return False, "no_target_element", None
        
        # 解析候选信息
        candidate_info = parse_candidate(target_element)
        bbox = candidate_info.get('bbox')
        
        # 用属性定位元素
        element, method = find_element_by_all_attributes(self._page, candidate_info, bbox=bbox)
        
        if element:
            print(f"    ✓ 找到元素 ({method})")
            try:
                self._page.evaluate("(el) => el.style.border='3px solid blue'", element)
            except:
                pass
            return True, f"success: {method}", element
        else:
            print(f"    ✗ 未找到元素")
            return False, "element_not_found", None
    
    def _verify_single_action(
        self,
        action: Action,
        annotation_id: str,
    ) -> Dict[str, Any]:
        """
        验证单个 Action
        
        Args:
            action: Action 对象
            annotation_id: Record 的 annotation_id
            
        Returns:
            验证结果字典
        """
        action_uid = action.metadata.get('action_uid', '')
        operation = action.metadata.get('operation', {})
        
        result = {
            'action_idx': action.action_idx,
            'action_uid': action_uid,
            'action_type': action.action_type,
            'action_repr': action.action_repr,
            'mhtml_found': False,
            'coord_success': False,
            'attr_success': False,
            'coord_reason': '',
            'attr_reason': '',
        }
        
        # 查找 MHTML 文件
        mhtml_path = self._find_mhtml_path(annotation_id, action_uid)
        if not mhtml_path:
            result['coord_reason'] = 'mhtml_not_found'
            result['attr_reason'] = 'mhtml_not_found'
            return result
        
        result['mhtml_found'] = True
        
        # 加载 MHTML 页面
        file_url = f'file://{os.path.abspath(mhtml_path)}'
        try:
            self._page.goto(file_url, wait_until='domcontentloaded', timeout=self.timeout)
            time.sleep(0.3)
        except Exception as e:
            result['coord_reason'] = f'load_failed: {str(e)[:50]}'
            result['attr_reason'] = f'load_failed: {str(e)[:50]}'
            return result
        
        action_repr = action.action_repr or f"{action.action_type} action"
        print(f"\n  操作: {action_repr}")
        
        # 坐标定位验证（使用内化方法）
        print("\n[指标1] 坐标定位:")
        try:
            coord_success, coord_reason, _ = self._verify_by_coords(action)
            result['coord_success'] = coord_success
            result['coord_reason'] = coord_reason
        except Exception as e:
            result['coord_reason'] = f'exception: {str(e)[:50]}'
        
        # 属性定位验证（使用内化方法）
        print("\n[指标2] 属性定位:")
        try:
            attr_success, attr_reason, _ = self._verify_by_attrs(action)
            result['attr_success'] = attr_success
            result['attr_reason'] = attr_reason
        except Exception as e:
            result['attr_reason'] = f'exception: {str(e)[:50]}'
        
        return result
    
    def check(self, record: Record) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        检查单个 Record 的静态可执行性
        
        Args:
            record: GUI Agent Record
            
        Returns:
            (errors, warnings, stats) 元组
        """
        errors = []
        warnings = []
        
        # 获取 annotation_id
        annotation_id = record.metadata.get('annotation_id', '')
        if not annotation_id:
            errors.append("Missing annotation_id in record metadata")
            return errors, warnings, {
                'total_actions': len(record.actions),
                'verified_actions': 0,
                'coord_success': 0,
                'attr_success': 0,
                'coord_rate': 0.0,
                'attr_rate': 0.0,
                'action_results': [],
            }
        
        # 确保浏览器已启动
        self._ensure_browser()
        
        # 验证每个 Action
        action_results = []
        mhtml_found_count = 0
        coord_success_count = 0
        attr_success_count = 0
        
        for action in record.actions:
            result = self._verify_single_action(action, annotation_id)
            action_results.append(result)
            
            if result['mhtml_found']:
                mhtml_found_count += 1
                if result['coord_success']:
                    coord_success_count += 1
                if result['attr_success']:
                    attr_success_count += 1
            else:
                warnings.append(
                    f"MHTML not found for action {action.action_idx}: {action.metadata.get('action_uid', '')[:8]}..."
                )
        
        # 计算成功率
        if mhtml_found_count > 0:
            coord_rate = coord_success_count / mhtml_found_count
            attr_rate = attr_success_count / mhtml_found_count
        else:
            coord_rate = 0.0
            attr_rate = 0.0
            errors.append("No MHTML files found for any action")
        
        stats = {
            'total_actions': len(record.actions),
            'verified_actions': mhtml_found_count,
            'coord_success': coord_success_count,
            'attr_success': attr_success_count,
            'coord_rate': coord_rate,
            'attr_rate': attr_rate,
            'action_results': action_results,
        }
        
        return errors, warnings, stats
    
    def check_batch(
        self,
        records: List[Record],
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        批量检查多个 Record
        
        Args:
            records: Record 列表
            verbose: 是否打印进度
            
        Returns:
            批量统计结果
        """
        all_results = []
        total_verified = 0
        total_coord = 0
        total_attr = 0
        
        try:
            for idx, record in enumerate(records):
                if verbose:
                    print(f"[{idx+1}/{len(records)}] {record.metadata.get('annotation_id', '')[:8]}...", end=" ")
                
                errors, warnings, stats = self.check(record)
                all_results.append({
                    'sample_id': record.sample_id,
                    'annotation_id': record.metadata.get('annotation_id', ''),
                    'website': record.website,
                    'errors': errors,
                    'warnings': warnings,
                    'stats': stats,
                })
                
                total_verified += stats['verified_actions']
                total_coord += stats['coord_success']
                total_attr += stats['attr_success']
                
                if verbose:
                    v = stats['verified_actions']
                    if v > 0:
                        print(f"坐标: {stats['coord_success']}/{v} ({stats['coord_rate']:.0%}) | "
                              f"属性: {stats['attr_success']}/{v} ({stats['attr_rate']:.0%})")
                    else:
                        print("⚠ 无可验证动作")
        
        finally:
            # 批量完成后关闭浏览器
            self._close_browser()
        
        # 汇总统计
        return {
            'total_records': len(records),
            'total_verified': total_verified,
            'total_coord_success': total_coord,
            'total_attr_success': total_attr,
            'coord_rate': total_coord / total_verified if total_verified > 0 else 0.0,
            'attr_rate': total_attr / total_verified if total_verified > 0 else 0.0,
            'results': all_results,
        }
    
    def __del__(self):
        """析构时关闭浏览器"""
        self._close_browser()


# =============================================================================
# 注册检查器
# =============================================================================

register_static_checker('mind2web', Mind2WebStaticChecker)


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """命令行测试入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mind2Web 静态可执行性检查')
    parser.add_argument('--data-path', type=str, 
                        default='/mnt/petrelfs/liuhaoze/datasets/Agent_Data/Mind2Web/data',
                        help='Mind2Web 数据路径')
    parser.add_argument('--raw-dump', type=str,
                        default=Mind2WebStaticChecker.DEFAULT_RAW_DUMP_PATH,
                        help='raw_dump 数据集路径')
    parser.add_argument('--batch', type=int, default=5,
                        help='批量测试记录数')
    parser.add_argument('--show', action='store_true',
                        help='显示浏览器窗口')
    args = parser.parse_args()
    
    # 导入 loader
    from loaders import Mind2WebLoader
    
    print("=" * 60)
    print("Mind2Web 静态可执行性检查")
    print("=" * 60)
    print(f"数据路径: {args.data_path}")
    print(f"Raw Dump: {args.raw_dump}")
    print(f"测试数量: {args.batch}")
    print()
    
    # 加载数据
    loader = Mind2WebLoader(args.data_path)
    loader.load()
    
    # 创建检查器
    checker = Mind2WebStaticChecker(
        raw_dump_path=args.raw_dump,
        headless=not args.show,
    )
    
    # 获取前 N 条记录
    records = []
    for i, record in enumerate(loader.iterate()):
        if i >= args.batch:
            break
        records.append(record)
    
    print(f"加载了 {len(records)} 条记录")
    print()
    
    # 批量检查
    batch_stats = checker.check_batch(records, verbose=True)
    
    # 打印汇总
    print()
    print("=" * 60)
    print("汇总结果")
    print("=" * 60)
    print(f"测试记录数: {batch_stats['total_records']}")
    print(f"可验证动作数: {batch_stats['total_verified']}")
    print(f"坐标定位成功率: {batch_stats['total_coord_success']}/{batch_stats['total_verified']} "
          f"({batch_stats['coord_rate']:.1%})")
    print(f"属性定位成功率: {batch_stats['total_attr_success']}/{batch_stats['total_verified']} "
          f"({batch_stats['attr_rate']:.1%})")


if __name__ == '__main__':
    main()
