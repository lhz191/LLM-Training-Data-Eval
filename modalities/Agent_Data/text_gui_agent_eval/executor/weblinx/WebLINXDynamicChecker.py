#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX 动态可执行性检查器

在真实网站上执行 Action 序列，验证是否可以成功执行。

与静态可执行性的区别：
- 静态：使用 pages/*.html 快照，通过 uid (data-webtasks-id) 定位
- 动态：使用真实网站，通过属性 (tag, class, text 等) 定位（真实网站无 uid）
"""

import os
import sys
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
    get_element_info,
    verify_by_coords,
    verify_by_attrs,
)

try:
    from playwright.sync_api import sync_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    print("⚠️ Playwright not installed. Install with: pip install playwright && playwright install")


class WebLINXDynamicChecker:
    """
    WebLINX 动态可执行性检查器
    
    在真实网站上执行 Action 序列，验证是否可以成功执行。
    
    验证方式：
    1. 打开 record.metadata['website_url'] 对应的真实网站
    2. 尝试用属性定位元素（真实网站无 data-webtasks-id）
    3. 执行操作并验证结果
    
    Args:
        headless: 是否使用无头浏览器模式
        timeout: 页面加载超时时间（毫秒）
    """
    
    def __init__(
        self,
        headless: bool = True,
        timeout: int = 60000,
    ):
        if not HAS_PLAYWRIGHT:
            raise ImportError("Playwright is required. Install with: pip install playwright && playwright install")
        
        self.headless = headless
        self.timeout = timeout
        self._playwright = None
        self._browser = None
        self._page = None
    
    def _ensure_browser(self):
        """确保浏览器已启动"""
        if self._page is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-site-isolation-trials",
                ]
            )
            context = self._browser.new_context(
                viewport={"width": DEFAULT_VIEWPORT_WIDTH, "height": DEFAULT_VIEWPORT_HEIGHT},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            self._page = context.new_page()
            # 反检测脚本
            self._page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            """)
    
    def _close_browser(self):
        """关闭浏览器"""
        if self._page:
            self._page = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
    
    def _dismiss_overlays(self):
        """尝试关闭页面上常见的遮罩层、弹窗、Cookie 同意框等"""
        print("  检查并关闭遮罩层/弹窗...")
        page = self._page
        
        close_selectors = [
            "button[id*='accept']", "button[id*='cookie']",
            "button[class*='accept']", "button[class*='cookie']", "button[class*='consent']",
            "[aria-label*='accept']", "[aria-label*='Accept']",
            "[aria-label*='close']", "[aria-label*='Close']",
            "button[class*='close']", "button[class*='dismiss']",
            "#onetrust-accept-btn-handler", ".onetrust-close-btn-handler",
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
        
        overlay_selectors = [
            "[class*='overlay']", "[class*='modal']", "[class*='backdrop']",
            "[class*='popup']", "[class*='cookie']", "[class*='consent']",
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
                            if box and box['width'] > DEFAULT_VIEWPORT_WIDTH * 0.5:
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
        action,
        element=None,
    ) -> Tuple[bool, str]:
        """
        执行单个操作
        
        Args:
            action: Action 对象
            element: 预先定位的元素
            
        Returns:
            (success, reason)
        """
        page = self._page
        action_type = action.action_type
        action_value = action.action_value
        
        try:
            # say: 不需要 DOM 操作
            if action_type == 'say':
                print(f"    [SAY] {action_value}")
                return True, "say_action"
            
            # load: 导航到 URL
            if action_type == 'load':
                url = action_value
                if url:
                    print(f"    [LOAD] {url}")
                    page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
                    time.sleep(3)
                    return True, "loaded"
                return False, "no_url"
            
            # scroll: 滚动页面
            if action_type == 'scroll':
                scroll_value = action_value or "0,0"
                try:
                    x, y = map(float, scroll_value.split(','))
                    page.evaluate(f"window.scrollTo({x}, {y})")
                    print(f"    [SCROLL] ({x}, {y})")
                    return True, "scrolled"
                except:
                    return False, "invalid_scroll_value"
            
            # 需要元素的操作
            if element is None:
                return False, "no_element"
            
            # 滚动到元素
            try:
                element.scroll_into_view_if_needed()
            except:
                pass
            time.sleep(0.3)
            
            # click
            if action_type == 'click':
                element.click()
                print(f"    [CLICK] ✓")
                time.sleep(1)
                return True, "clicked"
            
            # text_input
            if action_type == 'text_input':
                element.click()
                time.sleep(0.2)
                page.keyboard.type(action_value or "")
                print(f"    [TEXT_INPUT] {action_value}")
                time.sleep(0.5)
                return True, "typed"
            
            # change (select)
            if action_type == 'change':
                try:
                    element.select_option(label=action_value)
                    print(f"    [CHANGE] {action_value}")
                    return True, "changed"
                except:
                    element.click()
                    time.sleep(0.5)
                    option = page.locator(f"text={action_value}").first
                    option.click(timeout=3000)
                    return True, "changed_by_click"
            
            # submit
            if action_type == 'submit':
                element.click()
                print(f"    [SUBMIT] ✓")
                time.sleep(1)
                return True, "submitted"
            
            return False, f"unknown_action_type: {action_type}"
        
        except Exception as e:
            return False, f"execution_error: {str(e)[:100]}"
    
    def check(
        self,
        record,
        execute: bool = True,
        max_actions: int = None,
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        在真实网站上验证并执行 Record 的 action 序列
        
        Args:
            record: Record 对象
            execute: 是否执行操作
            max_actions: 最多执行多少个操作
        """
        errors = []
        warnings = []
        
        demo_id = record.metadata.get('demo_id', '')
        website = record.website or 'N/A'
        website_url = record.metadata.get('website_url', '')
        utterances = record.metadata.get('full_utterances', '')
        actions = record.actions
        
        # 确定起始 URL
        start_url = website_url
        if not start_url:
            # 尝试从第一个 load action 获取
            for a in actions:
                if a.action_type == 'load' and a.action_value:
                    start_url = a.action_value
                    break
        
        if not start_url:
            errors.append("无法确定起始 URL")
            return errors, warnings, {
                'total_actions': len(actions),
                'attr_success': 0,
                'exec_success': 0,
                'action_results': [],
            }
        
        print("=" * 80)
        print(f"WebLINX 动态验证")
        print("=" * 80)
        print(f"Demo ID: {demo_id}")
        print(f"网站: {website}")
        print(f"URL: {start_url}")
        print(f"操作数: {len(actions)}")
        print(f"视口: {DEFAULT_VIEWPORT_WIDTH} x {DEFAULT_VIEWPORT_HEIGHT}")
        print("=" * 80)
        
        self._ensure_browser()
        page = self._page
        
        # 打开网站
        print(f"\n打开 {start_url}...")
        try:
            page.goto(start_url, timeout=self.timeout, wait_until="domcontentloaded")
        except Exception as e:
            print(f"  ⚠ 页面加载超时: {e}")
            errors.append(f"page_load_timeout: {e}")
        
        time.sleep(5)
        self._dismiss_overlays()
        
        # 验证每个操作
        results = []
        num_actions = min(len(actions), max_actions) if max_actions else len(actions)
        
        # 统计计数
        uid_required_count = 0  # 需要定位的操作数
        coord_success_count = 0
        attr_success_count = 0
        exec_success_count = 0
        
        for i in range(num_actions):
            action = actions[i]
            action_type = action.action_type
            action_value = action.action_value or ''
            target_uid = action.target_element
            action_repr = action.action_repr or f"{action_type}({action_value})"
            
            print(f"\n{'─' * 60}")
            print(f"步骤 {i+1}/{num_actions}: [{action_type.upper()}]")
            print(f"操作描述: {action_repr}")
            
            result = {
                'step': i,
                'action_idx': action.action_idx,
                'action_type': action_type,
                'action_value': action_value,
                'target_uid': target_uid,
                'coord_success': False,
                'coord_reason': '',
                'attr_success': False,
                'attr_reason': '',
                'exec_success': False,
                'exec_reason': '',
            }
            
            # 不需要定位的操作（不计入坐标/属性定位统计）
            if action_type not in UID_REQUIRED_ACTIONS:
                result['coord_reason'] = 'no_uid_required'
                result['attr_reason'] = 'no_uid_required'
                # 注意：不设置 coord_success/attr_success，因为不需要定位
                
                # 显示具体操作类型
                if action_type == 'say':
                    print(f"    💬 代理回复: \"{action_value}\"")
                elif action_type == 'load':
                    print(f"    🌐 加载页面: {action_value or 'N/A'}")
                elif action_type == 'scroll':
                    print(f"    📜 滚动: {action_value or 'N/A'}")
                else:
                    print(f"    ⏭️ 跳过定位（{action_type} 不需要 uid）")
                
                if execute:
                    exec_ok, exec_reason = self._execute_action(action, element=None)
                    result['exec_success'] = exec_ok
                    result['exec_reason'] = exec_reason
                    if exec_ok:
                        exec_success_count += 1
                        print(f"    ✅ 执行成功")
                
                results.append(result)
                continue
            
            # 需要定位的操作
            uid_required_count += 1
            print(f"target_uid: {target_uid or '(无)'}")
            
            # 从 candidates 获取元素属性
            candidates = action.candidates or []
            element_data = None
            
            if target_uid and candidates:
                element_data = find_candidate_by_uid(target_uid, candidates)
            
            if not element_data and candidates:
                element_data = parse_weblinx_candidate(candidates[0]) if candidates else None
            
            coord_element = None
            attr_element = None
            
            if element_data:
                bbox = element_data.get('bbox')
                expected_tag = element_data.get('tag', '')
                
                # ===== 坐标定位 =====
                print(f"\n[指标1] 坐标定位:")
                if bbox:
                    print(f"    [坐标] bbox=({bbox.get('x', 0):.1f}, {bbox.get('y', 0):.1f}), size={bbox.get('width', 0):.0f}x{bbox.get('height', 0):.0f}, tag={expected_tag}")
                    coord_ok, coord_reason, coord_info, coord_element = verify_by_coords(
                        page, element_data, scroll_x=0, scroll_y=0, verbose=True
                    )
                    result['coord_success'] = coord_ok
                    result['coord_reason'] = coord_reason
                    if coord_ok:
                        coord_success_count += 1
                        print(f"    ✓ 坐标定位成功")
                    else:
                        print(f"    ✗ 坐标定位失败: {coord_reason}")
                else:
                    result['coord_reason'] = 'no_bbox'
                    print(f"    ✗ 无 bbox（数据集缺失）")
                
                # ===== 属性定位 =====
                print(f"\n[指标2] 属性定位:")
                attr_ok, attr_reason, attr_info, attr_element = verify_by_attrs(
                    page, element_data, bbox=bbox, verbose=True
                )
                result['attr_success'] = attr_ok
                result['attr_reason'] = attr_reason
                if attr_ok:
                    attr_success_count += 1
            else:
                result['coord_reason'] = 'no_candidate_data'
                result['attr_reason'] = 'no_candidate_data'
                print(f"    ❌ 无 candidate 数据（找不到匹配的 uid）")
            
            # 执行操作（优先用坐标定位的元素）
            if execute:
                exec_element = coord_element if result['coord_success'] else (attr_element if result['attr_success'] else None)
                if exec_element:
                    exec_ok, exec_reason = self._execute_action(action, element=exec_element)
                    result['exec_success'] = exec_ok
                    result['exec_reason'] = exec_reason
                    if exec_ok:
                        exec_success_count += 1
                        print(f"\n✅ 执行成功: {exec_reason}")
                    else:
                        print(f"\n❌ 执行失败: {exec_reason}")
                else:
                    result['exec_reason'] = 'no_element'
                    print(f"\n❌ 无法执行（两种定位均失败）")
            
            results.append(result)
        
        # 统计结果
        print("\n" + "=" * 80)
        print("验证结果汇总")
        print("=" * 80)
        
        total = len(results)
        # 坐标/属性定位成功率：只统计需要定位的操作
        coord_rate = coord_success_count / uid_required_count if uid_required_count > 0 else 0
        attr_rate = attr_success_count / uid_required_count if uid_required_count > 0 else 0
        # 执行成功率：统计所有操作
        exec_rate = exec_success_count / total if total > 0 else 0
        
        print(f"\n需要定位的操作: {uid_required_count}/{total}")
        print(f"坐标定位成功: {coord_success_count}/{uid_required_count} ({100*coord_rate:.1f}%)")
        print(f"属性定位成功: {attr_success_count}/{uid_required_count} ({100*attr_rate:.1f}%)")
        if execute:
            print(f"执行成功: {exec_success_count}/{total} ({100*exec_rate:.1f}%)")
        
        return errors, warnings, {
            'demo_id': demo_id,
            'website': website,
            'url': start_url,
            'total_actions': total,
            'uid_required_actions': uid_required_count,
            # 与 compute_dynamic_executability 期望的键名一致
            'coords_success': coord_success_count,
            'coords_rate': coord_rate,
            'attrs_success': attr_success_count,
            'attrs_rate': attr_rate,
            'executed_actions': exec_success_count,
            'execution_rate': exec_rate,
            'action_results': results,
        }
    
    def __del__(self):
        self._close_browser()
