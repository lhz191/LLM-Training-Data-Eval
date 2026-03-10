#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多轮 Text-GUI Agent Trustworthy

将整个 Session 的所有轮次拼成一条完整轨迹，喂给 Guard 模型做 session-level 安全评估。
不展平——模型需要看到跨轮上下文才能检测渐进式攻击。
"""

import json
import time
from datetime import datetime
from typing import Iterator, Dict, Any, Optional, List
from collections import Counter

import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
_mt_dir = os.path.dirname(_this_dir)
_agent_data_dir = os.path.dirname(_mt_dir)
_st_dir = os.path.join(_agent_data_dir, 'text_gui_agent_eval')

if _mt_dir not in sys.path:
    sys.path.insert(0, _mt_dir)
if _st_dir not in sys.path:
    sys.path.insert(0, _st_dir)

from data_types import Session, Record  # noqa: E402

_project_root = os.path.dirname(os.path.dirname(_agent_data_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluator import AgentDoGEvaluator  # noqa: E402
from common.base import BaseGuardEvaluator  # noqa: E402


def compute_trustworthy(
    session_iterator: Iterator[Session],
    dataset_name: str = "Unknown",
    evaluator: Optional[BaseGuardEvaluator] = None,
    model_path: Optional[str] = None,
    output_file: Optional[str] = None,
    max_sessions: Optional[int] = None,
    progress_interval: int = 10,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Session-level 安全评估：每个 Session 的所有轮次拼成一条完整轨迹。
    """
    start_time = time.time()

    if evaluator is None:
        if model_path is None:
            raise ValueError("必须提供 evaluator 或 model_path 参数")
        evaluator = AgentDoGEvaluator(model_path)

    model_name = evaluator.model_name
    model_info = getattr(evaluator, 'model_path', model_name)

    print(f"\n{'='*70}")
    print(f"Trustworthy Evaluation — Multi-Turn Text-GUI Agent ({model_name.upper()})")
    print(f"{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"模式: session-level（完整多轮轨迹）")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {model_info}")
    if max_sessions:
        print(f"Session 限制: {max_sessions}")
    print()

    total = 0
    safe_count = 0
    unsafe_count = 0
    total_rounds = 0

    risk_source_counter = Counter()
    failure_mode_counter = Counter()
    real_world_harm_counter = Counter()

    session_results: List[Dict[str, Any]] = []
    unsafe_sessions: List[Dict[str, Any]] = []

    for session in session_iterator:
        if max_sessions and total >= max_sessions:
            break

        total += 1
        total_rounds += len(session.rounds)

        try:
            result = evaluator.evaluate_session(
                rounds=session.rounds,
                session_id=session.session_id,
            )
            session_results.append(result)

            if result.get('is_safe', True):
                safe_count += 1
            else:
                unsafe_count += 1
                unsafe_sessions.append(result)

                if result.get('risk_source'):
                    risk_source_counter[result['risk_source']] += 1
                if result.get('failure_mode'):
                    failure_mode_counter[result['failure_mode']] += 1
                if result.get('real_world_harm'):
                    real_world_harm_counter[result['real_world_harm']] += 1

        except Exception as e:
            print(f"  Warning: 评估 session {session.session_id} 时出错: {e}")
            continue

        if show_progress and total % progress_interval == 0:
            safe_rate = safe_count / total if total > 0 else 0
            print(f"  [{total}] 安全率: {100*safe_rate:.1f}% ({safe_count}/{total})")

    elapsed = time.time() - start_time
    safe_rate = safe_count / total if total > 0 else 0

    results = {
        'dataset': dataset_name,
        'guard_model': model_name,
        'model_path': str(model_info),
        'evaluation_mode': 'session-level',
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'total_sessions': total,
        'total_rounds': total_rounds,
        'avg_rounds_per_session': total_rounds / total if total > 0 else 0,
        'safe_count': safe_count,
        'unsafe_count': unsafe_count,
        'safe_rate': safe_rate,
        'risk_source_distribution': dict(risk_source_counter.most_common()),
        'failure_mode_distribution': dict(failure_mode_counter.most_common()),
        'real_world_harm_distribution': dict(real_world_harm_counter.most_common()),
        'unsafe_sessions': unsafe_sessions,
    }

    print()
    print(f"{'='*70}")
    print(f"【Trustworthy 评估结果 (Session-Level)】")
    print(f"{'='*70}")
    print(f"  Guard 模型: {model_name}")
    print(f"  总 Session 数: {total:,}")
    if total > 0:
        print(f"  总轮次数: {total_rounds:,} (平均 {total_rounds/total:.1f} 轮/session)")
    print(f"  安全 Session: {safe_count:,} ({100*safe_rate:.1f}%)")
    print(f"  不安全 Session: {unsafe_count:,} ({100*(1-safe_rate):.1f}%)")
    if elapsed > 0:
        print(f"  耗时: {elapsed:.1f}s ({total/elapsed:.2f} sessions/s)")

    if unsafe_count > 0:
        print()
        print(f"【风险来源分布 (Top 5)】")
        for source, count in risk_source_counter.most_common(5):
            ratio = count / unsafe_count
            print(f"    - {source}: {count} ({100*ratio:.1f}%)")

        print()
        print(f"【失效模式分布 (Top 5)】")
        for mode, count in failure_mode_counter.most_common(5):
            ratio = count / unsafe_count
            print(f"    - {mode}: {count} ({100*ratio:.1f}%)")

        print()
        print(f"【现实危害分布 (Top 5)】")
        for harm, count in real_world_harm_counter.most_common(5):
            ratio = count / unsafe_count
            print(f"    - {harm}: {count} ({100*ratio:.1f}%)")

    print(f"{'='*70}")

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {output_file}")

    return results
