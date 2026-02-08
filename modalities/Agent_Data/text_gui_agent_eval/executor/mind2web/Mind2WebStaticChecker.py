#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mind2Web 静态可执行性检查器
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional

# 确保父目录在 path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from text_gui_executor import StaticExecutabilityChecker
from data_types import Record, Action

from .constants import VIEWPORT_WIDTH, VIEWPORT_HEIGHT
from .utils import (
    parse_candidate,
    verify_by_coords,
    verify_by_attrs,
)

try:
    from playwright.sync_api import sync_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    print("⚠️ Playwright not installed. Install with: pip install playwright && playwright install")

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
    
    def _highlight_element(self, element, color: str = 'green', duration: float = 0.3):
        """高亮元素（调试用）"""
        try:
            self._page.evaluate(f"""(el) => {{
                el.style.outline = '3px solid {color}';
                el.style.outlineOffset = '2px';
            }}""", element)
            if not self.headless:
                time.sleep(duration)
        except:
            pass
    
    def _execute_action(self, action: Action, element) -> Tuple[bool, str]:
        """
        执行操作（与 DynamicChecker 逻辑一致）
        
        Args:
            action: Action 对象
            element: Playwright 元素句柄
            
        Returns:
            (success, reason)
        """
        page = self._page
        operation = action.metadata.get('operation', {})
        op = operation.get('op', '').upper()
        value = operation.get('value', '')
        
        try:
            # 滚动到元素
            try:
                element.scroll_into_view_if_needed()
            except:
                pass
            time.sleep(0.5)
            
            if op == 'CLICK':
                element.click(force=True)  # force=True 跳过静态 HTML 的可点击检查
                print(f"    ✓ 点击成功")
                time.sleep(0.5)  # 静态不需要太长等待
                return True, "success"
            
            elif op == 'HOVER':
                element.hover()
                print(f"    ✓ 悬停成功")
                time.sleep(0.5)
                return True, "success"
            
            elif op == 'TYPE':
                element.click(force=True)
                time.sleep(0.3)
                # 选中已有文本
                try:
                    page.evaluate("(el) => { if(el.select) el.select(); }", element)
                except:
                    pass
                page.keyboard.type(value)
                print(f"    ✓ 输入成功: {value}")
                time.sleep(0.5)
                return True, "success"
            
            elif op == 'SELECT':
                element.click(force=True)
                time.sleep(0.5)
                try:
                    tag = page.evaluate("(el) => el.tagName.toLowerCase()", element)
                    if tag == 'select':
                        element.select_option(label=value)
                    else:
                        option = page.locator(f"text={value}").first
                        option.click(timeout=3000)
                    print(f"    ✓ 选择成功: {value}")
                    time.sleep(0.5)
                    return True, "success"
                except Exception as e:
                    return False, f"select_error: {e}"
            
            else:
                return False, f"unknown_op: {op}"
        
        except Exception as e:
            return False, f"execution_error: {e}"
    
    def _verify_single_action(
        self,
        action: Action,
        annotation_id: str,
        execute: bool = True,
    ) -> Dict[str, Any]:
        """
        验证单个 Action
        
        Args:
            action: Action 对象
            annotation_id: Record 的 annotation_id
            execute: 是否执行操作
            
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
            print(f"    ❌ MHTML 文件未找到: {action_uid}")
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
            error_msg = str(e)
            print(f"    ❌ 页面加载失败: {error_msg}")
            result['coord_reason'] = f'load_failed: {error_msg}'
            result['attr_reason'] = f'load_failed: {error_msg}'
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
        
        # 选择用于执行的元素（优先坐标定位，其次属性定位）
        element = coord_element if coord_success else (attr_element if attr_success else None)
        
        # 高亮元素
        if element:
            color = 'green' if coord_success else 'blue'
            self._highlight_element(element, color=color, duration=0.3)
        
        # 执行操作
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
            result['executed'] = False
            result['exec_reason'] = 'element_not_found'
            print(f"\n❌ 无法执行（定位失败）")
        
        return result
    
    def check(self, record: Record, execute: bool = True) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        检查单个 Record 的静态可执行性
        
        Args:
            record: GUI Agent Record
            execute: 是否执行操作（默认 True）
            
        Returns:
            (errors, warnings, stats) 元组，warnings 固定为空列表
        """
        errors = []
        
        # 获取 annotation_id
        annotation_id = record.metadata.get('annotation_id', '')
        if not annotation_id:
            errors.append("Missing annotation_id in record metadata")
            return errors, [], {
                'total_actions': len(record.actions),
                'verified_actions': 0,
                'coord_success': 0,
                'attr_success': 0,
                'executed_success': 0,
                'coord_rate': 0.0,
                'attr_rate': 0.0,
                'exec_rate': 0.0,
                'action_results': [],
            }
        
        # 确保浏览器已启动
        self._ensure_browser()
        
        # 打印 Record 信息
        total_actions = len(record.actions)
        print(f"\n{'='*70}")
        print(f"📋 Record: {record.sample_id} | annotation_id: {annotation_id}")
        print(f"   网站: {record.website or 'N/A'} | Actions: {total_actions}")
        print(f"   执行模式: {'开启' if execute else '关闭'}")
        print(f"{'='*70}")
        
        # 验证每个 Action
        action_results = []
        mhtml_found_count = 0
        coord_success_count = 0
        attr_success_count = 0
        exec_success_count = 0
        
        for idx, action in enumerate(record.actions):
            print(f"\n{'─'*60}")
            action_uid = action.metadata.get('action_uid', '')
            print(f"步骤 {idx+1}/{total_actions}: [{action.action_type.upper()}] {action_uid}")
            result = self._verify_single_action(action, annotation_id, execute=execute)
            action_results.append(result)
            
            if result['mhtml_found']:
                mhtml_found_count += 1
                if result['coord_success']:
                    coord_success_count += 1
                else:
                    errors.append(f"Action[{idx}]: coord_failed ({result.get('coord_reason', '?')})")
                    
                if result['attr_success']:
                    attr_success_count += 1
                else:
                    errors.append(f"Action[{idx}]: attr_failed ({result.get('attr_reason', '?')})")
                
                if result.get('executed'):
                    exec_success_count += 1
            else:
                reason = result.get('coord_reason', 'mhtml_not_found')
                errors.append(f"Action[{idx}]: {reason}")
        
        # 计算成功率
        if mhtml_found_count > 0:
            coord_rate = coord_success_count / mhtml_found_count
            attr_rate = attr_success_count / mhtml_found_count
            exec_rate = exec_success_count / mhtml_found_count
        else:
            coord_rate = 0.0
            attr_rate = 0.0
            exec_rate = 0.0
            errors.append("No MHTML files found for any action")
        
        stats = {
            'total_actions': len(record.actions),
            'verified_actions': mhtml_found_count,
            'coord_success': coord_success_count,
            'attr_success': attr_success_count,
            'executed_success': exec_success_count,
            'coord_rate': coord_rate,
            'attr_rate': attr_rate,
            'exec_rate': exec_rate,
            'action_results': action_results,
        }
        
        # 打印汇总
        print(f"\n{'='*70}")
        print(f"汇总: 坐标定位 {coord_success_count}/{mhtml_found_count} ({coord_rate:.1%})")
        print(f"      属性定位 {attr_success_count}/{mhtml_found_count} ({attr_rate:.1%})")
        if execute:
            print(f"      执行成功 {exec_success_count}/{mhtml_found_count} ({exec_rate:.1%})")
        print(f"{'='*70}")
        
        return errors, [], stats
    
    def __del__(self):
        """析构时关闭浏览器"""
        self._close_browser()


# =============================================================================
# 动态可执行性检查器
# =============================================================================

