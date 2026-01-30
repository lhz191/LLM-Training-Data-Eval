#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX 静态可执行性检查器
"""

import os
import re
import json
import time
from typing import Dict, List, Tuple, Optional, Any

from .constants import (
    UID_REQUIRED_ACTIONS,
    DEFAULT_VIEWPORT_WIDTH,
    DEFAULT_VIEWPORT_HEIGHT,
)
from .utils import (
    parse_weblinx_candidate,
    find_candidate_by_uid,
    build_css_selector,
    verify_weblinx_element_match,
)

try:
    from playwright.sync_api import sync_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    print("⚠️ Playwright not installed. Install with: pip install playwright && playwright install")


class WebLINXStaticChecker:
    """
    WebLINX 静态可执行性检查器
    
    类似 Mind2Web，使用 pages/*.html 快照验证 Action 是否可定位到目标元素。
    
    验证方式：
    1. 通过 demo + turn 找到对应的 page 文件
    2. 用 Playwright 加载 HTML 页面
    3. 验证 uid (data-webtasks-id) 能否被定位
    
    Args:
        raw_data_path: WebLINX raw_data 目录路径（包含 demonstrations/）
        headless: 是否使用无头浏览器模式
        timeout: 页面加载超时时间（秒）
    """
    
    # 默认 raw_data 路径
    DEFAULT_RAW_DATA_PATH = '/home/liuhaoze/Downloads/raw_data'
    
    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        headless: bool = True,
        timeout: int = 30,
    ):
        if not HAS_PLAYWRIGHT:
            raise ImportError("Playwright is required. Install with: pip install playwright && playwright install")
        
        self.raw_data_path = raw_data_path or self.DEFAULT_RAW_DATA_PATH
        self.headless = headless
        self.timeout = timeout * 1000  # 转换为毫秒
        
        # 浏览器实例（延迟初始化）
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        
        # replay 缓存（避免重复加载）
        self._replay_cache: Dict[str, dict] = {}
    
    def _ensure_browser(self):
        """确保浏览器已启动"""
        if self._page is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            context = self._browser.new_context(
                viewport={'width': DEFAULT_VIEWPORT_WIDTH, 'height': DEFAULT_VIEWPORT_HEIGHT}
            )
            self._page = context.new_page()
    
    def _set_viewport(self, action):
        """根据 action 的 viewport 字段设置页面大小"""
        viewport_str = action.metadata.get('viewport', '')
        
        # 解析 viewport 字符串，格式如 "714h x 1536w"
        width, height = DEFAULT_VIEWPORT_WIDTH, DEFAULT_VIEWPORT_HEIGHT
        if viewport_str:
            height_match = re.search(r'(\d+)h', viewport_str)
            width_match = re.search(r'(\d+)w', viewport_str)
            if height_match and width_match:
                height = int(height_match.group(1))
                width = int(width_match.group(1))
        
        try:
            self._page.set_viewport_size({'width': width, 'height': height})
        except Exception:
            pass  # 忽略设置 viewport 失败的情况
    
    def _close_browser(self):
        """关闭浏览器"""
        if self._page:
            self._page.close()
            self._page = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
    
    def _load_replay(self, demo_name: str) -> Optional[dict]:
        """加载指定 demo 的 replay.json（带缓存）"""
        if demo_name in self._replay_cache:
            return self._replay_cache[demo_name]
        
        replay_path = os.path.join(self.raw_data_path, 'demonstrations', demo_name, 'replay.json')
        if not os.path.exists(replay_path):
            return None
        
        try:
            with open(replay_path) as f:
                replay = json.load(f)
            self._replay_cache[demo_name] = replay
            return replay
        except Exception as e:
            print(f"⚠️ Failed to load replay.json for {demo_name}: {e}")
            return None
    
    def _get_page_path(self, demo_name: str, turn_idx: int) -> Optional[str]:
        """获取指定 turn 对应的 page 文件路径"""
        replay = self._load_replay(demo_name)
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
        
        page_path = os.path.join(self.raw_data_path, 'demonstrations', demo_name, 'pages', page)
        if os.path.exists(page_path):
            return page_path
        return None
    
    def _get_scroll_info(self, demo_name: str, turn_idx: int) -> Tuple[float, float]:
        """
        从 replay.json 获取滚动信息
        
        WebLINX 的 bbox 是视口坐标，需要通过 pageY - clientY 计算滚动偏移
        
        Args:
            demo_name: demo ID
            turn_idx: turn 索引
            
        Returns:
            (scroll_x, scroll_y) 滚动偏移，如果无法获取则返回 (0, 0)
        """
        replay = self._load_replay(demo_name)
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
    
    def _verify_by_coords(
        self, 
        element_data: dict = None,
        demo_name: str = None,
        turn_idx: int = -1,
    ) -> Tuple[bool, str, dict, any]:
        """
        通过坐标定位验证元素（类似 Mind2Web）
        
        WebLINX 的 bbox 是视口坐标，需要先从 replay.json 获取滚动偏移，
        滚动到正确位置后再用 elementsFromPoint 定位。
        
        注意：这是独立的指标，不涉及 UID 验证！
        
        Args:
            element_data: 元素信息（来自 parse_weblinx_candidate，包含 bbox、tag、class 等）
            demo_name: demo ID（用于获取滚动信息）
            turn_idx: turn 索引（用于获取滚动信息）
            
        Returns:
            (success, reason, element_info, element_handle)
        """
        if not element_data:
            return False, "no_element_data", {}, None
        
        # 获取 bbox
        bbox = element_data.get('bbox')
        if not bbox:
            return False, "no_bbox", {}, None
        
        # 从 replay.json 获取滚动信息
        scroll_x, scroll_y = self._get_scroll_info(demo_name, turn_idx)
        
        # 先滚动到数据收集时的位置
        if scroll_y != 0 or scroll_x != 0:
            try:
                self._page.evaluate(f"window.scrollTo({scroll_x}, {scroll_y})")
                self._page.wait_for_timeout(300)  # 等待滚动完成
                print(f"    [滚动] 已滚动到 scrollY={scroll_y:.0f} (从 replay.json 获取)")
            except Exception as e:
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
                all_elements = self._page.evaluate(f"""() => {{
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
                    element = self._page.evaluate_handle(f"""() => {{
                        const elements = document.elementsFromPoint({target_x}, {target_y});
                        return elements[{elem_idx}];
                    }}""")
                    
                    # 检查 null 或 undefined
                    is_invalid = self._page.evaluate("(el) => el === null || el === undefined || !el", element)
                    if is_invalid:
                        continue
                    
                    last_top_element = element
                    
                    # 使用 verify_weblinx_element_match 验证属性是否匹配
                    if element_data:
                        is_match, reason, matched, total = verify_weblinx_element_match(self._page, element, element_data)
                        if is_match:
                            print(f"    [{point_name}] 第{elem_idx}层元素直接匹配成功")
                            print(f"    ✓ 匹配成功 ({matched}/{total}): {reason}")
                            try:
                                self._page.evaluate("(el) => el.style.border='3px solid green'", element)
                            except:
                                pass
                            element_info = self._get_element_info(element)
                            return True, f"success@{point_name}_layer{elem_idx} ({matched}/{total})", element_info, element
                    
                    # 搜索子元素（如果有期望的 tag）
                    if expected_tag:
                        children_info = self._page.evaluate(f"""(el) => {{
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
                        
                        best_child = self._page.evaluate_handle(f"""(el) => {{
                            const children = el.querySelectorAll('{expected_tag}');
                            return children[{best_match_idx}];
                        }}""", element)
                        
                        best_rect = self._page.evaluate("""(el) => {
                            const rect = el.getBoundingClientRect();
                            return {x: rect.x, y: rect.y, width: rect.width, height: rect.height};
                        }""", best_child)
                        
                        size_diff = abs(best_rect['width'] - expected_w) + abs(best_rect['height'] - expected_h)
                        
                        if size_diff > 5:
                            continue
                        
                        if element_data:
                            is_match, reason, matched, total = verify_weblinx_element_match(self._page, best_child, element_data)
                            if is_match:
                                print(f"    [{point_name}] 第{elem_idx}层的子元素匹配成功")
                                print(f"    ✓ 匹配成功 ({matched}/{total}): {reason}")
                                try:
                                    self._page.evaluate("(el) => el.style.border='3px solid green'", best_child)
                                except:
                                    pass
                                element_info = self._get_element_info(best_child)
                                return True, f"success_child@{point_name}_layer{elem_idx}[{best_match_idx}] ({matched}/{total})", element_info, best_child
            
            # 所有点都失败
            if last_top_element:
                top_info = self._page.evaluate("""(el) => ({
                    tag: el.tagName.toLowerCase(),
                    rect: el.getBoundingClientRect()
                })""", last_top_element)
                print(f"    ✗ 3个检测点都未找到匹配元素")
                print(f"    [顶层元素] <{top_info['tag']}> @ ({top_info['rect']['x']:.0f},{top_info['rect']['y']:.0f})")
                try:
                    self._page.evaluate("(el) => el.style.border='3px solid orange'", last_top_element)
                except:
                    pass
                element_info = self._get_element_info(last_top_element)
                return False, "no_match_all_points", element_info, last_top_element
            else:
                print(f"    ✗ 未找到元素")
                return False, "element_not_found_at_coords", {}, None
                
        except Exception as e:
            print(f"    ✗ 坐标定位错误: {e}")
            return False, f"coord_error: {str(e)}", {}, None
    
    def _get_element_info(self, element) -> dict:
        """获取元素的详细信息（不截断任何字段）"""
        try:
            info = self._page.evaluate("""(el) => {
                if (!el || !el.tagName) return {};
                const rect = el.getBoundingClientRect();
                return {
                    tag: el.tagName.toLowerCase(),
                    uid: el.getAttribute('data-webtasks-id') || '',
                    id: el.id || '',
                    className: el.className || '',
                    text: (el.textContent || '').replace(/\\s+/g, ' ').trim(),
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
    
    def _verify_by_attrs(
        self, 
        element_data: dict = None, 
        bbox: dict = None,
        demo_name: str = None,
        turn_idx: int = -1,
    ) -> Tuple[bool, str, dict, any]:
        """
        通过属性定位验证元素
        
        定位策略：
        1. 用 CSS 选择器找元素
        2. 逐个验证所有属性（包括 xpath 和 text）
        3. 找到第一个验证通过的就返回成功
        
        Args:
            element_data: 元素信息（来自 parse_weblinx_candidate）
            bbox: 目标元素的边界框（用于多匹配时选择最近的）
            demo_name: demo ID（用于获取滚动信息）
            turn_idx: turn 索引（用于获取滚动信息）
            
        Returns:
            (success, reason, element_info, element_handle)
        """
        if not element_data:
            print(f"    ❌ 无元素信息（数据集缺失）")
            return False, "no_element_data", {}, None
        
        # element_data 已经是解析后的格式（来自 parse_weblinx_candidate）
        tag_name = element_data.get('tag', '')
        
        # 获取 bbox（用于多匹配时选择）
        if bbox is None:
            bbox = element_data.get('bbox')
        
        # 如果有 bbox 用于坐标筛选，需要先滚动到正确位置
        if bbox and demo_name and turn_idx >= 0:
            scroll_x, scroll_y = self._get_scroll_info(demo_name, turn_idx)
            if scroll_y != 0 or scroll_x != 0:
                try:
                    self._page.evaluate(f"window.scrollTo({scroll_x}, {scroll_y})")
                    self._page.wait_for_timeout(300)
                except:
                    pass
        
        # 用 CSS 选择器找元素
        selector, desc = build_css_selector(element_data)
        
        if not selector:
            print(f"    ✗ 没有可用属性构建 CSS 选择器")
            return False, "no_css_selector", {'tag': tag_name}, None
        
        # 简化过长的选择器显示
        if len(selector) > 80:
            print(f"    [CSS选择器] {desc}")
        else:
            print(f"    [CSS选择器] {selector}")
        
        try:
            elements = self._page.query_selector_all(selector)
        except Exception as e:
            print(f"    ✗ CSS 选择器错误: {str(e)}")
            return False, f"css_error: {str(e)}", {'tag': tag_name}, None
        
        if not elements:
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
        if len(elements) > 1:
            print(f"    找到 {len(elements)} 个候选，按坐标距离排序验证")
        
        # 逐个验证，找到第一个通过验证的
        fail_reasons = []  # 收集失败原因
        for i, element in enumerate(elements):
            is_match, reason, matched, total = verify_weblinx_element_match(self._page, element, element_data)
            
            if is_match:
                element_info = self._get_element_info(element)
                if len(elements) == 1:
                    print(f"    ✓ 匹配成功 ({matched}/{total}): {reason}")
                else:
                    print(f"    ✓ 第{i+1}个元素匹配成功 ({matched}/{total}): {reason}")
                return True, f"match ({matched}/{total})", element_info, element
            else:
                # 记录失败原因
                fail_reasons.append((i + 1, reason, matched, total))
        
        # 所有元素都验证失败
        element_info = self._get_element_info(elements[0]) if elements else {}
        print(f"    ✗ 所有 {len(elements)} 个候选元素验证失败")
        # 打印每个元素的失败原因
        for idx, reason, matched, total in fail_reasons:
            print(f"      - 元素{idx}: ({matched}/{total}) {reason}")
        return False, "all_verify_failed", element_info, None
    
    def _verify_uid_in_page(self, uid: str) -> Tuple[bool, str, dict, any]:
        """
        在当前加载的页面中验证 uid 是否可定位
        
        Returns:
            (success, reason, element_info, element_handle)
        """
        if not uid:
            return False, "no_uid", {}, None
        
        selector = f'[data-webtasks-id="{uid}"]'
        
        try:
            elements = self._page.query_selector_all(selector)
            
            if not elements:
                return False, "uid_not_found", {}, None
            
            if len(elements) > 1:
                # 多个匹配，取第一个
                element = elements[0]
                reason = f"found_multiple({len(elements)})"
            else:
                element = elements[0]
                reason = "found"
            
            # 获取元素信息（不截断任何字段）
            element_info = {}
            try:
                element_info['tag'] = self._page.evaluate('(el) => el.tagName', element)
                element_info['visible'] = element.is_visible()
                rect = element.bounding_box()
                if rect:
                    element_info['bbox'] = rect
                # 获取更多属性
                attrs = self._page.evaluate("""(el) => {
                    return {
                        text: (el.textContent || '').replace(/\\s+/g, ' ').trim(),
                        id: el.id || '',
                        className: el.className || '',
                        type: el.type || '',
                        placeholder: el.placeholder || '',
                        value: el.value || '',
                    };
                }""", element)
                element_info.update(attrs)
            except:
                pass
            
            return True, reason, element_info, element
            
        except Exception as e:
            return False, f"error: {str(e)[:50]}", {}, None
    
    def _highlight_element(self, element, color: str = 'green', duration: float = 0.5):
        """
        高亮元素
        
        Args:
            element: Playwright 元素句柄
            color: 边框颜色 ('green', 'orange', 'blue', 'red')
            duration: 高亮持续时间（秒）
        """
        if not element:
            return
        
        try:
            # 添加高亮边框
            self._page.evaluate(f"(el) => {{ el.style.outline = '3px solid {color}'; el.style.outlineOffset = '2px'; }}", element)
            
            # 滚动到元素可见位置
            try:
                element.scroll_into_view_if_needed()
            except:
                pass
            
            # 非无头模式下等待一下让用户看到
            if not self.headless:
                time.sleep(duration)
        except:
            pass
    
    def _execute_action(self, action, element) -> Tuple[bool, str]:
        """
        执行操作（点击、输入等）
        
        Args:
            action: Action 对象
            element: Playwright 元素句柄
            
        Returns:
            (success, reason)
        """
        action_type = action.action_type
        action_value = action.action_value
        
        try:
            # 先滚动到元素位置
            try:
                element.scroll_into_view_if_needed(timeout=2000)
            except:
                pass
            
            if action_type == 'click':
                # 使用 force=True 跳过可点击检查（静态 HTML 可能有遮挡）
                element.click(timeout=5000, force=True)
                return True, "click_executed"
            
            elif action_type == 'text_input':
                if action_value:
                    element.fill(action_value, timeout=5000, force=True)
                    return True, "text_input_executed"
                else:
                    return False, "no_text_value"
            
            elif action_type == 'change':
                if action_value:
                    try:
                        element.select_option(value=action_value, timeout=5000)
                        return True, "change_executed"
                    except:
                        # 如果 select_option 失败，尝试直接点击
                        element.click(timeout=5000, force=True)
                        return True, "change_click_fallback"
                else:
                    return False, "no_change_value"
            
            elif action_type == 'submit':
                element.press('Enter', timeout=5000)
                return True, "submit_executed"
            
            else:
                return True, f"no_execution_needed({action_type})"
                
        except Exception as e:
            return False, f"execution_error: {str(e)}"  # 不截断错误信息
    
    def _verify_single_action(
        self,
        action,
        demo_name: str,
        execute: bool = True,
    ) -> Dict[str, Any]:
        """
        验证单个 Action - 三指标验证（UID、坐标、属性）
        
        数据来源：action.candidates（训练数据，属性可能被截断）
        
        Args:
            action: Action 对象
            demo_name: demo ID
            execute: 是否执行操作
        """
        turn_idx = action.metadata.get('turn', -1)
        action_type = action.action_type
        target_uid = action.target_element  # WebLINX 中是 uid 字符串
        
        result = {
            'action_idx': action.action_idx,
            'turn': turn_idx,
            'action_type': action_type,
            'action_repr': action.action_repr,
            'target_uid': target_uid,
            'page_found': False,
            # 三个定位指标
            'uid_success': False,
            'uid_reason': '',
            'coord_success': False,
            'coord_reason': '',
            'attr_success': False,
            'attr_reason': '',
            # 其他
            'element_info': {},
            'executed': False,
            'exec_reason': '',
            'data_source': 'candidates',  # 只使用 candidates
        }
        
        # 打印操作描述（与 Mind2Web 格式一致）
        action_repr = action.action_repr or f"{action_type} action"
        print(f"操作描述: {action_repr}")
        
        # 如果不需要 uid 的操作，标记为成功
        if action_type not in UID_REQUIRED_ACTIONS:
            result['uid_success'] = True
            result['uid_reason'] = 'no_uid_required'
            result['coord_success'] = True
            result['coord_reason'] = 'no_uid_required'
            result['attr_success'] = True
            result['attr_reason'] = 'no_uid_required'
            
            # 特殊处理 say 操作，展示说话内容
            if action_type == 'say':
                speaker_match = re.search(r'speaker="([^"]*)"', action.action_repr or '')
                utterance_match = re.search(r'utterance="([^"]*)"', action.action_repr or '')
                
                speaker = speaker_match.group(1) if speaker_match else 'unknown'
                utterance = utterance_match.group(1) if utterance_match else action.action_value or ''
                
                if speaker == 'instructor':
                    print(f"    💬 用户说: \"{utterance}\"")
                elif speaker == 'navigator':
                    print(f"    🤖 代理回复: \"{utterance}\"")
                else:
                    print(f"    💭 {speaker}: \"{utterance}\"")
            elif action_type == 'load':
                print(f"    🌐 加载页面: {action.action_value or 'N/A'}")
            elif action_type == 'scroll':
                print(f"    📜 滚动: {action.action_value or 'N/A'}")
            else:
                print(f"    ⏭️ 跳过定位（{action_type} 不需要 uid）")
            
            return result
        
        if not target_uid:
            result['uid_reason'] = 'no_target_uid'
            result['coord_reason'] = 'no_target_uid'
            result['attr_reason'] = 'no_target_uid'
            print(f"    ❌ uid 为空（click(uid=None)）")
            return result
        
        # 获取 page 文件路径
        page_path = self._get_page_path(demo_name, turn_idx)
        if not page_path:
            result['uid_reason'] = 'page_not_found'
            result['coord_reason'] = 'page_not_found'
            result['attr_reason'] = 'page_not_found'
            print(f"    ❌ 页面文件未找到（turn={turn_idx}）")
            return result
        
        result['page_found'] = True
        result['page_file'] = os.path.basename(page_path)
        
        # 设置 viewport（每个 action 可能有不同的 viewport）
        self._set_viewport(action)
        
        # 加载页面
        file_url = f'file://{os.path.abspath(page_path)}'
        try:
            self._page.goto(file_url, wait_until='domcontentloaded', timeout=self.timeout)
        except Exception as e:
            error_msg = str(e)
            result['uid_reason'] = f'load_error: {error_msg}'
            result['coord_reason'] = f'load_error: {error_msg}'
            result['attr_reason'] = f'load_error: {error_msg}'
            print(f"    ❌ 页面加载失败: {error_msg}")
            return result
        
        # 从 candidates（训练数据）中获取元素信息
        target_candidate = find_candidate_by_uid(action.candidates, target_uid)
        if target_candidate:
            element_data = parse_weblinx_candidate(target_candidate)
            result['candidate_found'] = True
        else:
            element_data = None
            result['candidate_found'] = False
        
        bbox = element_data.get('bbox') if element_data else None
        
        # ===== 指标1: UID 定位（独立）=====
        print(f"\n[指标1] UID 定位:")
        uid_success, uid_reason, uid_element_info, uid_element = self._verify_uid_in_page(target_uid)
        result['uid_success'] = uid_success
        result['uid_reason'] = uid_reason
        result['element_info'] = uid_element_info
        if uid_success:
            if uid_element_info:
                tag = uid_element_info.get('tag', '?')
                elem_class = uid_element_info.get('className', '')
                class_str = f", class={elem_class}" if elem_class else ""
                print(f"    ✓ 找到元素: tag={tag}{class_str}")
            else:
                print(f"    ✓ 找到元素")
        else:
            print(f"    ✗ 未找到: {uid_reason}")
        
        # ===== 指标2: 坐标定位（独立，只验证属性匹配）=====
        print(f"\n[指标2] 坐标定位:")
        if bbox:
            expected_tag = element_data.get('tag', '') if element_data else ''
            print(f"    [坐标] bbox=({bbox.get('x', 0):.1f}, {bbox.get('y', 0):.1f}), size={bbox.get('width', 0):.0f}x{bbox.get('height', 0):.0f}, tag={expected_tag}")
        else:
            print(f"    ✗ 无 bbox（数据集缺失）")
        
        coord_success, coord_reason, coord_element_info, coord_element = self._verify_by_coords(
            element_data=element_data,  # 使用 candidates 数据
            demo_name=demo_name,         # 用于获取滚动信息
            turn_idx=turn_idx,           # 用于获取滚动信息
        )
        result['coord_success'] = coord_success
        result['coord_reason'] = coord_reason
        result['coord_element_info'] = coord_element_info
        
        # ===== 指标3: 属性定位（独立，CSS 选择器定位）=====
        print(f"\n[指标3] 属性定位:")
        attr_success, attr_reason, attr_element_info, attr_element = self._verify_by_attrs(
            element_data=element_data,  # 使用 candidates 数据
            bbox=bbox,
            demo_name=demo_name,         # 用于获取滚动信息（多匹配时坐标筛选）
            turn_idx=turn_idx,           # 用于获取滚动信息
        )
        result['attr_success'] = attr_success
        result['attr_reason'] = attr_reason
        result['attr_element_info'] = attr_element_info
        
        # 找到的元素（用于高亮和执行）
        element = uid_element or coord_element or attr_element
        element_info = uid_element_info or coord_element_info or attr_element_info
        
        # 高亮元素
        if element:
            color = 'green' if uid_success else ('orange' if coord_success or attr_success else 'red')
            self._highlight_element(element, color=color, duration=0.3)
        
        # 执行操作（优先使用 UID 定位的元素）
        if execute and element:
            exec_success, exec_reason = self._execute_action(action, element)
            result['executed'] = exec_success
            result['exec_reason'] = exec_reason
            
            if exec_success:
                print(f"\n✅ 执行成功: {exec_reason}")
                if not self.headless:
                    time.sleep(0.3)
            else:
                print(f"\n❌ 执行失败: {exec_reason}")
        elif execute and not element:
            print(f"\n❌ 无法执行（三种定位均失败）")
        
        # 保存元素信息（用于调试）
        if element_data:
            result['expected_element'] = {
                'tag': element_data.get('tag'),
                'bbox': element_data.get('bbox'),
                'class': element_data.get('class'),
                'id': element_data.get('id'),
                'data_source': result.get('data_source', 'unknown'),
            }
        
        return result
    
    def check(self, record, execute: bool = True) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        验证 Record 的所有 Action 在 HTML 快照中的可定位性
        
        Args:
            record: Record 对象（来自 WebLINXLoader）
            execute: 是否执行操作（点击、输入等）
            
        Returns:
            (errors, warnings, stats)
        """
        errors = []
        warnings = []
        
        demo_name = record.metadata.get('demo_id', '')
        website = record.website or 'N/A'
        actions = record.actions
        total_actions = len(actions)
        
        # 与 Mind2Web 格式一致的 Record 头部
        print(f"\n{'='*70}")
        print(f"📋 Record: {record.sample_id} | demo_id: {demo_name}")
        print(f"   网站: {website} | Actions: {total_actions}")
        print(f"{'='*70}")
        
        # 确保浏览器已启动
        self._ensure_browser()
        
        # 验证每个 action
        action_results = []
        
        # 统计计数器 - 三指标
        uid_required_count = 0   # 需要定位的 action 数
        page_found_count = 0     # 找到页面文件的数量
        uid_success_count = 0    # UID 定位成功
        coord_success_count = 0  # 坐标定位成功
        attr_success_count = 0   # 属性定位成功
        exec_success_count = 0   # 执行成功
        
        for i, action in enumerate(actions):
            # 与 Mind2Web 格式一致的 Action 头部
            print(f"\n{'─'*60}")
            print(f"步骤 {i+1}/{total_actions}: [{action.action_type.upper()}]")
            
            result = self._verify_single_action(action, demo_name, execute=execute)
            action_results.append(result)
            
            if action.action_type in UID_REQUIRED_ACTIONS:
                uid_required_count += 1
                uid_reason = result.get('uid_reason', '')
                
                # 根据 reason 判断错误类型
                if uid_reason == 'no_target_uid':
                    errors.append(f"Action[{i}]: uid_is_none ({action.action_type}(uid=None))")
                elif uid_reason == 'page_not_found':
                    errors.append(f"Action[{i}]: page_not_found (turn={result.get('turn', '?')})")
                elif result['page_found']:
                    page_found_count += 1
                    
                    # 统计三个定位指标
                    if result.get('uid_success'):
                        uid_success_count += 1
                    if result.get('coord_success'):
                        coord_success_count += 1
                    if result.get('attr_success'):
                        attr_success_count += 1
                    
                    # 检查是否有任何定位成功
                    any_success = result.get('uid_success') or result.get('coord_success') or result.get('attr_success')
                    
                    if any_success:
                        if result.get('executed'):
                            exec_success_count += 1
                        elif result.get('exec_reason'):
                            # 定位成功但执行失败
                            errors.append(f"Action[{i}]: execution_failed ({result.get('exec_reason', '?')})")
                    else:
                        # 三种定位都失败
                        errors.append(f"Action[{i}]: all_locate_failed (uid:{uid_reason})")
                else:
                    errors.append(f"Action[{i}]: page_not_found (turn={result.get('turn', '?')})")
        
        # 计算统计
        page_rate = page_found_count / uid_required_count if uid_required_count > 0 else 1.0
        uid_rate = uid_success_count / page_found_count if page_found_count > 0 else 0.0
        coord_rate = coord_success_count / page_found_count if page_found_count > 0 else 0.0
        attr_rate = attr_success_count / page_found_count if page_found_count > 0 else 0.0
        exec_rate = exec_success_count / uid_required_count if uid_required_count > 0 else 0.0
        
        # 按逻辑顺序统计
        stats = {
            'total_actions': len(actions),
            'uid_required_count': uid_required_count,  # 需要定位的 action 数
            'page_found_count': page_found_count,      # 找到页面文件的数量
            'page_rate': page_rate,
            # 三指标定位成功数
            'uid_success_count': uid_success_count,
            'uid_rate': uid_rate,
            'coord_success_count': coord_success_count,
            'coord_rate': coord_rate,
            'attr_success_count': attr_success_count,
            'attr_rate': attr_rate,
            # 执行
            'exec_success_count': exec_success_count,
            'exec_rate': exec_rate,
            'error_count': len(errors),
            'action_results': action_results,
        }
        
        # 与 Mind2Web 格式一致的汇总
        print(f"\n{'─'*60}")
        print(f"📊 验证汇总:")
        print(f"   需要定位的 action: {uid_required_count}")
        print(f"   找到页面文件: {page_found_count} ({page_rate:.1%})")
        print(f"   [指标1] UID 定位: {uid_success_count}/{page_found_count} ({uid_rate:.1%})")
        print(f"   [指标2] 坐标定位: {coord_success_count}/{page_found_count} ({coord_rate:.1%})")
        print(f"   [指标3] 属性定位: {attr_success_count}/{page_found_count} ({attr_rate:.1%})")
        if execute:
            print(f"   执行成功: {exec_success_count}/{uid_required_count} ({exec_rate:.1%})")
        if errors:
            print(f"   错误数: {len(errors)}")
        
        return errors, [], stats
    
    def __del__(self):
        """析构时关闭浏览器"""
        self._close_browser()
