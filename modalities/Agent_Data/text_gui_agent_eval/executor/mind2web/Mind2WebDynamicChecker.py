#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mind2Web 动态可执行性检查器
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from text_gui_executor import DynamicExecutabilityChecker
from data_types import Record, Action

from .constants import VIEWPORT_WIDTH, VIEWPORT_HEIGHT
from .utils import (
    parse_candidate,
    find_element_by_all_attributes,
    verify_element_match,
)

try:
    from playwright.sync_api import sync_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    print("⚠️ Playwright not installed.")

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
        
        website = record.website
        task = record.instruction or "N/A"
        actions = record.actions
        
        url = self._guess_url(website)
        
        if not url:
            errors.append("Cannot determine website URL")
            return errors, [], {
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
            errors.append(f"page_load_timeout: {e}")
        
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
                        errors.append(f"Action[{i}]: exec_failed ({exec_reason})")
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
        
        return errors, [], {
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

