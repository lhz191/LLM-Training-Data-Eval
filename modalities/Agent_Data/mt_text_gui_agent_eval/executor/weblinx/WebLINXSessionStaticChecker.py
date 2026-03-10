#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebLINX 多轮静态可执行性检查器

逐轮验证 Session 中每个 Record 的 Action 在 HTML 快照上的可定位性。
复用单轮 WebLINXStaticChecker 的验证逻辑，但以 Session 为单位组织结果。

验证方式（与单轮一致）：
1. 通过 demo + turn 找到对应的 page 文件
2. 用 Playwright 加载 HTML 页面
3. 验证 uid (data-webtasks-id) 能否被定位
4. 三指标验证：UID、坐标、属性
"""

import os
import sys
import importlib.util
from typing import Dict, List, Tuple, Optional, Any

_current_file = os.path.abspath(__file__)
_weblinx_dir = os.path.dirname(_current_file)
_executor_dir = os.path.dirname(_weblinx_dir)
_mt_eval_dir = os.path.dirname(_executor_dir)
_agent_data_dir = os.path.dirname(_mt_eval_dir)

if _mt_eval_dir not in sys.path:
    sys.path.insert(0, _mt_eval_dir)

from data_types import Session
from mt_executor import SessionStaticChecker

_st_dir = os.path.join(_agent_data_dir, 'text_gui_agent_eval')
_st_executor_weblinx = os.path.join(_st_dir, 'executor', 'weblinx')

if _st_dir not in sys.path:
    sys.path.insert(0, _st_dir)
if _st_executor_weblinx not in sys.path:
    sys.path.insert(0, _st_executor_weblinx)


class WebLINXSessionStaticChecker(SessionStaticChecker):
    """
    WebLINX 多轮静态可执行性检查器

    内部实例化单轮 WebLINXStaticChecker，逐轮调用 check()，
    然后聚合为 Session 级别的统计结果。

    Args:
        raw_data_path: WebLINX raw_data 目录路径
        headless: 是否使用无头浏览器模式
        timeout: 页面加载超时时间（秒）
    """

    def __init__(
        self,
        raw_data_path: Optional[str] = None,
        headless: bool = True,
        timeout: int = 30,
    ):
        from executor.weblinx import WebLINXStaticChecker
        self._single_checker = WebLINXStaticChecker(
            raw_data_path=raw_data_path,
            headless=headless,
            timeout=timeout,
        )

    def check_session(
        self,
        session: Session,
        execute: bool = True,
    ) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        验证 Session 中所有轮次的静态可执行性

        Args:
            session: 多轮会话
            execute: 是否执行操作

        Returns:
            (errors, warnings, stats) 元组
        """
        all_errors = []
        all_warnings = []
        round_results = []

        total_uid_required = 0
        total_uid_success = 0
        total_coord_success = 0
        total_attr_success = 0
        total_exec_success = 0
        total_page_found = 0

        print(f"\n{'=' * 70}")
        print(f"Session 静态可执行性检查: {session.session_id}")
        print(f"  轮次数: {len(session.rounds)}")
        print(f"  总 action 数: {session.total_actions}")
        print(f"{'=' * 70}")

        for r_idx, record in enumerate(session.rounds):
            print(f"\n--- Round {r_idx + 1}/{len(session.rounds)} ---")
            errors, warnings, stats = self._single_checker.check(record, execute=execute)

            all_errors.extend([f"Round[{r_idx}].{e}" for e in errors])
            all_warnings.extend([f"Round[{r_idx}].{w}" for w in warnings])

            total_uid_required += stats.get('uid_required_count', 0)
            total_uid_success += stats.get('uid_success_count', 0)
            total_coord_success += stats.get('coord_success_count', 0)
            total_attr_success += stats.get('attr_success_count', 0)
            total_exec_success += stats.get('exec_success_count', 0)
            total_page_found += stats.get('page_found_count', 0)

            round_results.append({
                'round_idx': r_idx,
                'instruction': record.instruction,
                'stats': stats,
                'num_errors': len(errors),
            })

        uid_rate = total_uid_success / total_page_found if total_page_found > 0 else 0.0
        coord_rate = total_coord_success / total_page_found if total_page_found > 0 else 0.0
        attr_rate = total_attr_success / total_page_found if total_page_found > 0 else 0.0
        exec_rate = total_exec_success / total_uid_required if total_uid_required > 0 else 0.0

        session_stats = {
            'session_id': session.session_id,
            'total_rounds': len(session.rounds),
            'total_actions': session.total_actions,
            'total_uid_required': total_uid_required,
            'total_page_found': total_page_found,
            'uid_success': total_uid_success,
            'uid_rate': uid_rate,
            'coord_success': total_coord_success,
            'coord_rate': coord_rate,
            'attr_success': total_attr_success,
            'attr_rate': attr_rate,
            'exec_success': total_exec_success,
            'exec_rate': exec_rate,
            'round_results': round_results,
        }

        print(f"\n{'=' * 70}")
        print(f"Session 汇总:")
        print(f"  UID 定位: {total_uid_success}/{total_page_found} ({uid_rate:.1%})")
        print(f"  坐标定位: {total_coord_success}/{total_page_found} ({coord_rate:.1%})")
        print(f"  属性定位: {total_attr_success}/{total_page_found} ({attr_rate:.1%})")
        if execute:
            print(f"  执行成功: {total_exec_success}/{total_uid_required} ({exec_rate:.1%})")
        print(f"{'=' * 70}")

        return all_errors, all_warnings, session_stats

    def __del__(self):
        if hasattr(self, '_single_checker'):
            del self._single_checker
