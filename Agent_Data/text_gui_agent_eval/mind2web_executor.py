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

from text_gui_executor import (
    StaticExecutabilityChecker, 
    DynamicExecutabilityChecker,
    FormatChecker,
    HTMLLocator,
    register_static_checker,
    register_dynamic_checker,
    register_format_checker,
    register_html_locator,
)
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


# =============================================================================
# 公共验证函数（可被静态/动态检查器复用）
# =============================================================================

def verify_by_coords(page, target_element) -> Tuple[bool, str, Any]:
    """
    坐标定位验证（公共函数，来自 verify_dynamic.py 的 verify_by_coords1）
    
    使用 bounding_box 坐标定位元素，验证是否能找到匹配的元素。
    
    Args:
        page: Playwright page 对象
        target_element: 目标元素字典（pos_candidate 格式）
        
    Returns:
        (success, reason, element)
    """
    if not target_element:
        return False, "no_target_element", None
    
    # 解析候选信息
    candidate_info = parse_candidate(target_element)
    bbox = candidate_info.get('bbox')
    expected_tag = candidate_info.get('tag', '').lower()
    
    if not bbox:
        return False, "no_bbox", None
    
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
            
            # 检查 null 或 undefined
            is_invalid = page.evaluate("(el) => el === null || el === undefined || !el", element)
            if is_invalid:
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


def verify_by_attrs(page, target_element) -> Tuple[bool, str, Any]:
    """
    属性定位验证（公共函数，来自 verify_dynamic.py 的 verify_by_attrs）
    
    使用 tag/class/id 等属性定位元素。
    
    Args:
        page: Playwright page 对象
        target_element: 目标元素字典（pos_candidate 格式）
        
    Returns:
        (success, reason, element)
    """
    if not target_element:
        return False, "no_target_element", None
    
    # 解析候选信息
    candidate_info = parse_candidate(target_element)
    bbox = candidate_info.get('bbox')
    
    # 用属性定位元素
    element, method = find_element_by_all_attributes(page, candidate_info, bbox=bbox)
    
    if element:
        print(f"    ✓ 找到元素 ({method})")
        try:
            page.evaluate("(el) => el.style.border='3px solid blue'", element)
        except:
            pass
        return True, f"success: {method}", element
    else:
        print(f"    ✗ 未找到元素")
        return False, "element_not_found", None


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
    # 添加保护检查，防止元素已被移除或失效
    try:
        actual_attrs = page.evaluate("""(element) => {
            if (!element || !element.tagName) {
                return null;
            }
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
        
        if actual_attrs is None:
            return False, "element_stale_or_invalid", 0, 0
    except Exception as e:
        return False, f"element_evaluate_error: {str(e)[:50]}", 0, 0
    
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
    # 本机路径
    DEFAULT_RAW_DUMP_PATH = '/home/liuhaoze/data/raw_dump'
    # 远程路径（集群）
    # DEFAULT_RAW_DUMP_PATH = '/mnt/petrelfs/liuhaoze/datasets/Agent_Data/Mind2Web/raw_dump'
    
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
    # 验证方法（调用公共函数）
    # =========================================================================
    
    def _verify_by_coords(self, action: Action) -> Tuple[bool, str, Any]:
        """坐标定位验证（调用公共函数）"""
        return verify_by_coords(self._page, action.target_element)
    
    def _verify_by_attrs(self, action: Action) -> Tuple[bool, str, Any]:
        """属性定位验证（调用公共函数）"""
        return verify_by_attrs(self._page, action.target_element)
    
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
            验证结果字典，包含详细的目标元素信息和验证结果
        """
        action_uid = action.metadata.get('action_uid', '')
        operation = action.metadata.get('operation', {})
        
        # 解析目标元素信息
        target_info = {}
        if action.target_element:
            candidate_info = parse_candidate(action.target_element)
            bbox = candidate_info.get('bbox')
            target_info = {
                'tag': candidate_info.get('tag', ''),
                'classes': candidate_info.get('classes', []),
                'id': candidate_info.get('id', ''),
                'text': candidate_info.get('text', ''),
                'bbox': bbox,
            }
        
        result = {
            'action_idx': action.action_idx,
            'action_uid': action_uid,
            'action_type': action.action_type,
            'action_repr': action.action_repr,
            'target_element': target_info,  # 目标元素详细信息
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
            result['coord_reason'] = f'load_failed: {str(e)}'
            result['attr_reason'] = f'load_failed: {str(e)}'
            return result
        
        action_repr = action.action_repr or f"{action.action_type} action"
        print(f"操作描述: {action_repr}")
        
        # 坐标定位验证（使用内化方法）
        print("\n[指标1] 坐标定位:")
        try:
            coord_success, coord_reason, coord_element = self._verify_by_coords(action)
            result['coord_success'] = coord_success
            result['coord_reason'] = coord_reason
            # 如果成功，记录找到的元素信息
            if coord_success and coord_element:
                try:
                    found_info = self._page.evaluate("""(el) => ({
                        tag: el.tagName.toLowerCase(),
                        class: el.getAttribute('class') || '',
                        id: el.getAttribute('id') || '',
                    })""", coord_element)
                    result['coord_found_element'] = found_info
                except:
                    pass
        except Exception as e:
            result['coord_reason'] = f'exception: {str(e)}'
        
        # 属性定位验证（使用内化方法）
        print("\n[指标2] 属性定位:")
        try:
            attr_success, attr_reason, attr_element = self._verify_by_attrs(action)
            result['attr_success'] = attr_success
            result['attr_reason'] = attr_reason
            # 如果成功，记录找到的元素信息
            if attr_success and attr_element:
                try:
                    found_info = self._page.evaluate("""(el) => ({
                        tag: el.tagName.toLowerCase(),
                        class: el.getAttribute('class') || '',
                        id: el.getAttribute('id') || '',
                    })""", attr_element)
                    result['attr_found_element'] = found_info
                except:
                    pass
        except Exception as e:
            result['attr_reason'] = f'exception: {str(e)}'
        
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
        
        # 打印 Record 信息
        total_actions = len(record.actions)
        print(f"\n{'='*70}")
        print(f"📋 Record: {record.sample_id} | annotation_id: {annotation_id[:16]}...")
        print(f"   网站: {record.website or 'N/A'} | Actions: {total_actions}")
        print(f"{'='*70}")
        
        # 验证每个 Action
        action_results = []
        mhtml_found_count = 0
        coord_success_count = 0
        attr_success_count = 0
        
        for idx, action in enumerate(record.actions):
            print(f"\n{'─'*60}")
            print(f"步骤 {idx+1}/{total_actions}: [{action.action_type.upper()}] {action.metadata.get('action_uid', '')[:8]}...")
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
    
    def __del__(self):
        """析构时关闭浏览器"""
        self._close_browser()


# =============================================================================
# 动态可执行性检查器
# =============================================================================

class Mind2WebDynamicChecker(DynamicExecutabilityChecker):
    """
    Mind2Web 动态可执行性检查器
    
    在真实网站上执行 Action 序列，验证是否可以成功执行。
    """
    
    def __init__(self, headless: bool = True, timeout: int = 60):
        if not HAS_PLAYWRIGHT:
            raise ImportError("Playwright is required")
        
        self.headless = headless
        self.timeout = timeout * 1000
        self._playwright = None
        self._browser = None
        self._page = None
    
    def _ensure_browser(self):
        """确保浏览器已启动（配置来自 verify_dynamic.py）"""
        if self._page is None:
            self._playwright = sync_playwright().start()
            # 启动浏览器（与官方一致的设置）
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",  # 反检测
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-site-isolation-trials",
                ]
            )
            # 创建上下文（与官方一致的视口大小）
            context = self._browser.new_context(
                viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            self._page = context.new_page()
            # 注入反检测脚本
            self._page.add_init_script("""
                // 隐藏 webdriver 标志
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                // 伪造 plugins
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                // 伪造语言
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            """)
    
    def _close_browser(self):
        """关闭浏览器"""
        if self._browser:
            self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    # 特殊 URL 映射（不符合 www.xxx.com 模式的网站）
    SPECIAL_URL_MAPPING = {
        'new.mta.info': 'https://www.mta.info/',
        'mta.info': 'https://www.mta.info/',
    }
    
    def _guess_url(self, website: str) -> str:
        """根据 website 名称猜测 URL"""
        if not website:
            return None
        
        website_lower = website.lower()
        
        # 1. 特殊映射
        if website_lower in self.SPECIAL_URL_MAPPING:
            return self.SPECIAL_URL_MAPPING[website_lower]
        
        # 2. 自动猜测
        if '.' in website:
            parts = website.split('.')
            if len(parts) == 2 and parts[1] in ['com', 'org', 'net', 'info', 'io', 'fm']:
                return f'https://www.{website}'
            else:
                return f'https://{website}.com'
        
        return f'https://www.{website}.com'
    
    def _dismiss_overlays(self):
        """尝试关闭页面上常见的遮罩层、弹窗、Cookie 同意框等"""
        print("  检查并关闭遮罩层/弹窗...")
        page = self._page
        
        # 常见的关闭按钮选择器
        close_selectors = [
            # Cookie 同意
            "button[id*='accept']",
            "button[id*='cookie']",
            "button[class*='accept']",
            "button[class*='cookie']",
            "button[class*='consent']",
            "a[id*='accept']",
            "[aria-label*='accept']",
            "[aria-label*='Accept']",
            "[aria-label*='close']",
            "[aria-label*='Close']",
            "[aria-label*='dismiss']",
            # 关闭按钮
            "button[class*='close']",
            "button[class*='dismiss']",
            ".close-button",
            ".modal-close",
            ".popup-close",
            # OneTrust Cookie Banner (常见)
            "#onetrust-accept-btn-handler",
            ".onetrust-close-btn-handler",
        ]
        
        closed_count = 0
        for selector in close_selectors:
            try:
                elements = page.query_selector_all(selector)
                for elem in elements:
                    if elem.is_visible():
                        try:
                            elem.click(timeout=1000)
                            closed_count += 1
                            time.sleep(0.5)
                        except:
                            pass
            except:
                pass
        
        # 尝试隐藏常见的遮罩层
        overlay_selectors = [
            "[class*='overlay']",
            "[class*='modal']",
            "[class*='curtain']",
            "[class*='backdrop']",
            "[class*='popup']",
            "[class*='cookie']",
            "[class*='consent']",
            "[id*='overlay']",
            "[id*='modal']",
            "[id*='popup']",
            "[id*='cookie']",
            "#onetrust-banner-sdk",
        ]
        
        hidden_count = 0
        for selector in overlay_selectors:
            try:
                elements = page.query_selector_all(selector)
                for elem in elements:
                    if elem.is_visible():
                        try:
                            box = elem.bounding_box()
                            if box and box['width'] > VIEWPORT_WIDTH * 0.5 and box['height'] > VIEWPORT_HEIGHT * 0.3:
                                page.evaluate("(el) => el.style.display='none'", elem)
                                hidden_count += 1
                        except:
                            pass
            except:
                pass
        
        if closed_count > 0 or hidden_count > 0:
            print(f"    已处理 {closed_count} 个弹窗, 隐藏 {hidden_count} 个遮罩层")
            time.sleep(1)
        else:
            print(f"    未发现需要处理的遮罩层")
    
    def _execute_action(
        self, 
        action: Action, 
        element=None, 
        use_coords_element: bool = True
    ) -> Tuple[bool, str]:
        """
        执行单个操作（照搬自 verify_dynamic.py 的 execute_action）
        
        Args:
            action: Action 对象
            element: 预先定位的元素（可选）
            use_coords_element: 如果没有传入 element，是否使用坐标定位
            
        Returns:
            (success, reason)
        """
        page = self._page
        target_element = action.target_element
        operation = action.metadata.get('operation', {})
        op = operation.get('op', '')
        value = operation.get('value', '')
        
        # 如果没有传入元素，需要先定位
        if element is None:
            if not target_element:
                return False, "no_target_element"
            
            candidate_info = parse_candidate(target_element)
            bbox = candidate_info.get('bbox')
            
            if use_coords_element and bbox:
                # 使用坐标定位
                center_x = bbox['x'] + bbox['width'] / 2
                center_y = bbox['y'] + bbox['height'] / 2
                scroll_y = max(0, bbox['y'] - 200)
                page.evaluate(f"window.scrollTo(0, {scroll_y})")
                time.sleep(0.3)
                scroll_top = page.evaluate("window.pageYOffset")
                viewport_y = center_y - scroll_top
                element = page.evaluate_handle(
                    "(args) => document.elementFromPoint(args.x, args.y)",
                    {"x": center_x, "y": viewport_y}
                )
                # 检查 null 或 undefined
                is_invalid = page.evaluate("(el) => el === null || el === undefined || !el", element)
                if is_invalid:
                    return False, "element_not_found_by_coords"
            else:
                # 使用属性定位
                element, method = find_element_by_all_attributes(page, candidate_info, bbox=bbox)
                if not element:
                    return False, "element_not_found_by_attrs"
        
        # 执行操作
        try:
            # 滚动到元素
            try:
                element.scroll_into_view_if_needed()
            except:
                pass
            time.sleep(0.5)
            
            if op == 'CLICK':
                element.click()
                print(f"    ✓ 点击成功")
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                except:
                    pass
                time.sleep(1)
                return True, "success"
            
            elif op == 'HOVER':
                element.hover()
                print(f"    ✓ 悬停成功")
                time.sleep(1)
                return True, "success"
            
            elif op == 'TYPE':
                element.click()
                time.sleep(0.3)
                # 选中已有文本
                try:
                    page.evaluate("(el) => { if(el.select) el.select(); }", element)
                except:
                    pass
                page.keyboard.type(value)
                print(f"    ✓ 输入成功: {value}")
                time.sleep(1)
                return True, "success"
            
            elif op == 'SELECT':
                element.click()
                time.sleep(1)
                try:
                    tag = page.evaluate("(el) => el.tagName.toLowerCase()", element)
                    if tag == 'select':
                        element.select_option(label=value)
                    else:
                        option = page.locator(f"text={value}").first
                        option.click(timeout=3000)
                    print(f"    ✓ 选择成功: {value}")
                    time.sleep(1)
                    return True, "success"
                except Exception as e:
                    return False, f"select_error: {e}"
            
            else:
                return False, f"unknown_op: {op}"
        
        except Exception as e:
            return False, f"execution_error: {e}"
    
    # =========================================================================
    # 主检查方法（照搬自 verify_dynamic.py 的 verify_record）
    # =========================================================================
    
    def check(
        self, 
        record, 
        execute: bool = True,
        max_actions: int = None,
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        在真实网站上验证并执行 Record 的 action 序列
        
        使用两个独立指标：
        - 指标1 (坐标定位): 验证网站是否变化（数据集时效性）
        - 指标2 (属性定位): 验证对 Agent 训练是否有用（数据集实用性）
        
        Args:
            record: Record 对象
            execute: 是否执行操作（默认 True）。如果只想验证不执行，设为 False
            max_actions: 最多执行多少个操作（默认 None 表示全部）
        """
        errors = []
        warnings = []
        
        website = record.website
        task = record.instruction or "N/A"
        actions = record.actions
        
        url = self._guess_url(website)
        
        if not url:
            errors.append("Cannot determine website URL")
            return errors, warnings, {
                'total_actions': len(actions),
                'coords_success': 0,
                'attrs_success': 0,
                'executed_actions': 0,
                'action_results': [],
            }
        
        print("=" * 80)
        print(f"Mind2Web 动态验证 (Playwright)")
        print("=" * 80)
        print(f"网站: {website}")
        print(f"URL: {url}")
        print(f"任务: {task}")
        print(f"操作数: {len(actions)}")
        print(f"视口: {VIEWPORT_WIDTH} x {VIEWPORT_HEIGHT} (与官方一致)")
        print(f"验证模式: 双指标独立验证")
        print(f"  - 指标1: 坐标定位 + 属性验证 (网站变化检测) [绿框=成功, 橙框=失败]")
        print(f"  - 指标2: 属性定位 (Agent 训练可用性) [蓝框]")
        print("=" * 80)
        
        self._ensure_browser()
        page = self._page
        
        # 打开网站
        print(f"\n打开 {url}...")
        try:
            page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
        except Exception as e:
            print(f"  ⚠ 页面加载超时，尝试继续: {e}")
            warnings.append(f"Page load timeout: {e}")
        
        time.sleep(10)  # 与原版一致
        self._dismiss_overlays()
        
        # 验证每个操作
        results = []
        num_actions = min(len(actions), max_actions) if max_actions else len(actions)
        
        for i in range(num_actions):
            action = actions[i]
            action_repr = action.action_repr or f"{action.action_type} action"
            action_uid = action.metadata.get('action_uid', 'N/A')
            
            operation = action.metadata.get('operation', {})
            op = operation.get('op', '')
            value = operation.get('value', '')
            
            # 输出格式与 verify_dynamic.py 对齐
            print(f"\n{'─' * 60}")
            print(f"步骤 {i+1}/{num_actions}: [{op}] {value}")
            print(f"action_uid: {action_uid}")
            print(f"操作描述: {action_repr}")
            
            # 打印数据集属性
            target_element = action.target_element
            if target_element:
                candidate_info = parse_candidate(target_element)
                bbox = candidate_info.get('bbox')
                available_attrs = []
                if candidate_info.get('tag'): available_attrs.append(f"tag={candidate_info['tag']}")
                if candidate_info.get('id'): available_attrs.append(f"id={candidate_info['id']}")
                if candidate_info.get('name'): available_attrs.append(f"name={candidate_info['name']}")
                if candidate_info.get('class'): available_attrs.append(f"class={candidate_info['class']}")
                if candidate_info.get('aria_label'): available_attrs.append(f"aria-label={candidate_info['aria_label']}")
                if candidate_info.get('placeholder'): available_attrs.append(f"placeholder={candidate_info['placeholder']}")
                if bbox: available_attrs.append(f"坐标=({bbox['x']:.0f},{bbox['y']:.0f},{bbox['width']:.0f}x{bbox['height']:.0f})")
                print(f"数据集属性: {', '.join(available_attrs) if available_attrs else '无'}")
            
            # ===== 指标1: 坐标定位 =====
            print(f"\n[指标1] 坐标定位 [绿框=成功, 橙框=失败]:")
            coords_success, coords_reason, coords_element = verify_by_coords(page, target_element)
            
            # ===== 指标2: 属性定位 =====
            print(f"\n[指标2] 属性定位 [蓝框]:")
            attrs_success, attrs_reason, attrs_element = verify_by_attrs(page, target_element)
            
            # 简洁的结果行
            coords_mark = "✓" if coords_success else "✗"
            attrs_mark = "✓" if attrs_success else "✗"
            print(f"  => 结果: 坐标 {coords_mark} | 属性 {attrs_mark}")
            
            # 构建详细的 result 字典
            result_entry = {
                'step': i,
                'action_idx': action.action_idx,
                'action_uid': action_uid,  # 不截断
                'action_type': op,
                'action_repr': action_repr,  # 不截断
                'op': op,
                'value': value,  # 不截断
                'coords_success': coords_success,
                'coords_reason': coords_reason,  # 不截断
                'attrs_success': attrs_success,
                'attrs_reason': attrs_reason,  # 不截断
                'executed': False,
                'exec_reason': None,
            }
            
            # 添加 target_element 详细信息
            if target_element:
                candidate_info = parse_candidate(target_element)
                bbox = candidate_info.get('bbox')
                result_entry['target_element'] = {
                    'tag': candidate_info.get('tag', ''),
                    'classes': candidate_info.get('class', '').split() if candidate_info.get('class') else [],
                    'id': candidate_info.get('id', ''),
                    'name': candidate_info.get('name', ''),
                    'text': candidate_info.get('text', ''),  # 不截断
                    'aria_label': candidate_info.get('aria_label', ''),
                    'placeholder': candidate_info.get('placeholder', ''),
                    'bbox': bbox if bbox else None,
                }
            
            # 添加坐标定位找到的元素信息
            if coords_element:
                try:
                    result_entry['coord_found_element'] = {
                        'tag': coords_element.evaluate("el => el.tagName.toLowerCase()"),
                        'class': coords_element.get_attribute('class') or '',
                        'id': coords_element.get_attribute('id') or '',
                    }
                except:
                    result_entry['coord_found_element'] = None
            
            # 添加属性定位找到的元素信息
            if attrs_element:
                try:
                    result_entry['attr_found_element'] = {
                        'tag': attrs_element.evaluate("el => el.tagName.toLowerCase()"),
                        'class': attrs_element.get_attribute('class') or '',
                        'id': attrs_element.get_attribute('id') or '',
                    }
                except:
                    result_entry['attr_found_element'] = None
            
            results.append(result_entry)
            
            # ===== 执行操作（可选） =====
            if execute:
                # 优先用坐标定位的元素执行，其次用属性定位的元素
                exec_element = coords_element if coords_success else (attrs_element if attrs_success else None)
                if exec_element:
                    print(f"\n  [执行操作]")
                    exec_success, exec_reason = self._execute_action(action, element=exec_element)
                    results[-1]['executed'] = exec_success
                    results[-1]['exec_reason'] = exec_reason
                    if not exec_success:
                        print(f"  ⚠ 执行失败: {exec_reason}")
                        warnings.append(f"Action {i} execution failed: {exec_reason}")
                else:
                    print(f"\n  [跳过执行] 两种方式都未找到元素")
                    results[-1]['exec_reason'] = "element_not_found"
        
        # ===== 统计结果 =====
        print("\n" + "=" * 80)
        print("验证结果汇总")
        print("=" * 80)
        
        total = len(results)
        coords_success_count = sum(1 for r in results if r['coords_success'])
        attrs_success_count = sum(1 for r in results if r['attrs_success'])
        executed_count = sum(1 for r in results if r.get('executed'))
        
        print(f"\n指标1 (坐标定位 - 网站变化): {coords_success_count}/{total} ({100*coords_success_count/total:.1f}%)" if total > 0 else "")
        print(f"指标2 (属性定位 - Agent可用): {attrs_success_count}/{total} ({100*attrs_success_count/total:.1f}%)" if total > 0 else "")
        if execute:
            print(f"执行成功: {executed_count}/{total} ({100*executed_count/total:.1f}%)" if total > 0 else "")
        
        return errors, warnings, {
            'total_actions': total,
            'coords_success': coords_success_count,
            'coords_rate': coords_success_count / total if total > 0 else 0.0,
            'attrs_success': attrs_success_count,
            'attrs_rate': attrs_success_count / total if total > 0 else 0.0,
            'executed_actions': executed_count,
            'execution_rate': executed_count / total if total > 0 else 0.0,
            'action_results': results,
            'website': website,
            'url': url,
        }
    
    def __del__(self):
        self._close_browser()


# =============================================================================
# 格式检查器
# =============================================================================

class Mind2WebFormatChecker(FormatChecker):
    """
    Mind2Web 数据格式检查器
    
    检查项：
    1. Record 级别
       - annotation_id 是否存在
       - instruction 是否存在且非空
       - actions 是否存在且非空
       
    2. Action 级别
       - action_uid 是否存在
       - target_element 是否存在
       - operation 是否存在（op 字段）
       - candidates 是否存在
       
    3. 数据一致性检查
       - target 是否在 candidates 中（通过 backend_node_id 匹配）
       - backend_node_id 是否在 cleaned_html 中可找到
    """
    
    def check(self, record: Record) -> Tuple[List[str], List[str]]:
        """检查 Mind2Web Record 的数据格式"""
        errors = []
        warnings = []  # 保留接口，但不使用
        
        # === 1. Record 级别检查 ===
        
        # annotation_id
        annotation_id = record.metadata.get('annotation_id', '')
        if not annotation_id:
            errors.append("Record missing 'annotation_id' in metadata")
        
        # instruction
        if not record.instruction or not record.instruction.strip():
            errors.append("Record has empty 'instruction'")
        
        # actions
        if not record.actions:
            errors.append("Record has no actions")
            return errors, warnings  # 无法继续检查 action 级别
        
        # === 2. Action 级别检查 ===
        for i, action in enumerate(record.actions):
            action_errors, _ = self._check_action(action, i)
            errors.extend(action_errors)
        
        return errors, warnings
    
    def _check_action(self, action: Action, idx: int) -> Tuple[List[str], List[str]]:
        """检查单个 Action 的格式"""
        errors = []
        warnings = []  # 保留接口，但不使用
        prefix = f"Action[{idx}]"
        
        # action_uid
        action_uid = action.metadata.get('action_uid', '')
        if not action_uid:
            errors.append(f"{prefix}: missing 'action_uid'")
        
        # target_element
        target = action.target_element
        if not target:
            errors.append(f"{prefix}: missing 'target_element'")
        else:
            # 检查 backend_node_id
            backend_node_id = target.get('backend_node_id')
            if not backend_node_id:
                errors.append(f"{prefix}: target_element missing 'backend_node_id'")
        
        # operation
        operation = action.metadata.get('operation', {})
        if not operation:
            errors.append(f"{prefix}: missing 'operation' in metadata")
        else:
            op = operation.get('op', '').upper()
            value = operation.get('value', '')
            
            if not op:
                errors.append(f"{prefix}: operation missing 'op' field")
            else:
                # 根据操作类型检查 value
                # CLICK: 不应该有 value
                # SELECT/TYPE: 必须有 value
                if op == 'CLICK':
                    if value and value.strip():
                        errors.append(f"{prefix}: CLICK should not have value, got '{value[:30]}'")
                elif op in ('SELECT', 'TYPE'):
                    if not value or not value.strip():
                        errors.append(f"{prefix}: {op} must have value")
        
        # candidates
        candidates = action.candidates
        if not candidates:
            errors.append(f"{prefix}: no candidates")
        else:
            # === 3. 数据一致性检查 ===
            # 检查 target 是否在 candidates 中
            if target:
                target_in_candidates = self._check_target_in_candidates(target, candidates)
                if not target_in_candidates:
                    errors.append(f"{prefix}: target not found in candidates")
        
        # cleaned_html
        if not action.cleaned_html:
            errors.append(f"{prefix}: empty 'cleaned_html'")
        else:
            # 检查 backend_node_id 是否在 cleaned_html 中可找到
            if target:
                backend_node_id = target.get('backend_node_id')
                if backend_node_id:
                    # 在 cleaned_html 中搜索 backend_node_id
                    # 可能的格式: backend_node_id="136" 或 data-backend-node-id="136" 或直接作为某个属性值
                    node_id_str = str(backend_node_id)
                    if node_id_str not in action.cleaned_html:
                        errors.append(f"{prefix}: backend_node_id not found in cleaned_html")
        
        return errors, warnings
    
    def _check_target_in_candidates(self, target: Dict, candidates: List[Dict]) -> bool:
        """检查 target 是否在 candidates 中（通过 backend_node_id 匹配）"""
        target_node_id = target.get('backend_node_id')
        if not target_node_id:
            return False
        
        for cand in candidates:
            if cand.get('backend_node_id') == target_node_id:
                return True
        
        return False


# =============================================================================
# HTML 定位器
# =============================================================================

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


# =============================================================================
# 注册检查器和定位器
# =============================================================================

register_static_checker('mind2web', Mind2WebStaticChecker)
register_dynamic_checker('mind2web', Mind2WebDynamicChecker)
register_format_checker('mind2web', Mind2WebFormatChecker)
register_html_locator('mind2web', Mind2WebLocator)


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """命令行测试入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mind2Web 静态可执行性检查')
    parser.add_argument('--data-path', type=str, 
                        default='/home/liuhaoze/Desktop/mind2web',
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
    
    # 逐个检查
    total_verified = 0
    total_coord = 0
    total_attr = 0
    
    try:
        for idx, record in enumerate(records):
            print(f"[{idx+1}/{len(records)}] {record.metadata.get('annotation_id', '')[:8]}...", end=" ")
            errors, warnings, stats = checker.check(record)
            
            total_verified += stats['verified_actions']
            total_coord += stats['coord_success']
            total_attr += stats['attr_success']
            
            v = stats['verified_actions']
            if v > 0:
                print(f"坐标: {stats['coord_success']}/{v} ({stats['coord_rate']:.0%}) | "
                      f"属性: {stats['attr_success']}/{v} ({stats['attr_rate']:.0%})")
            else:
                print("⚠ 无可验证动作")
    finally:
        checker._close_browser()
    
    # 打印汇总
    print()
    print("=" * 60)
    print("汇总结果")
    print("=" * 60)
    print(f"测试记录数: {len(records)}")
    print(f"可验证动作数: {total_verified}")
    coord_rate = total_coord / total_verified if total_verified > 0 else 0.0
    attr_rate = total_attr / total_verified if total_verified > 0 else 0.0
    print(f"坐标定位成功率: {total_coord}/{total_verified} ({coord_rate:.1%})")
    print(f"属性定位成功率: {total_attr}/{total_verified} ({attr_rate:.1%})")


if __name__ == '__main__':
    main()
