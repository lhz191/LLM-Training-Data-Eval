#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Turn Diversity

在单轮六维度基础上，增加 session-level 多样性分析。

=== 设计 ===
1. 提取所有 Record，调用单轮 compute_diversity 获取六维度基础指标
2. 如果存在多轮 session (rounds > 1)，额外计算 session-level 指标：
   - 轮次数分布多样性
   - 轮间语义转移多样性
   - 轮间长度模式多样性

=== 与单轮 diversity 的关系 ===
单轮 diversity 衡量 record 之间的差异（action 类型、轨迹序列、页面结构等）。
多轮 diversity 额外衡量 session 内部结构的差异。
两者互补，不重复。
"""

import os
import sys
import json
import importlib.util
from typing import Iterator, Dict, Any, List, Optional
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_mt_dir = os.path.dirname(_this_dir)
_agent_data_dir = os.path.dirname(_mt_dir)
_st_dir = os.path.join(_agent_data_dir, 'text_gui_agent_eval')

# ---------------------------------------------------------------------------
# 从多轮 data_types 导入 Session / Record
# ---------------------------------------------------------------------------
if _mt_dir not in sys.path:
    sys.path.insert(0, _mt_dir)
from data_types import Session, Record  # noqa: E402

# ---------------------------------------------------------------------------
# 显式加载单轮 diversity 模块（避免和本文件同名冲突）
# ---------------------------------------------------------------------------
_st_div_path = os.path.join(_st_dir, 'metrics', 'diversity.py')
_spec = importlib.util.spec_from_file_location('st_diversity', _st_div_path)
_st_diversity = importlib.util.module_from_spec(_spec)

if _st_dir not in sys.path:
    sys.path.append(_st_dir)
_spec.loader.exec_module(_st_diversity)

compute_diversity = _st_diversity.compute_diversity
_compute_entropy = _st_diversity._compute_entropy


# ============================================================================
# 主入口
# ============================================================================

def compute_diversity_multiturn(
    session_iterator: Iterator[Session],
    dataset_name: str = "Unknown",
    output_file: Optional[str] = None,
    max_sessions: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    计算多轮数据集的多样性指标

    内部提取所有 Record 调用单轮 compute_diversity，
    然后追加 session-level 多样性分析。
    """
    sessions: List[Session] = []
    all_records: List[Record] = []
    for session in session_iterator:
        if max_sessions and len(sessions) >= max_sessions:
            break
        sessions.append(session)
        all_records.extend(session.rounds)

    has_multi_turn = any(s.is_multi_turn for s in sessions)

    base_results = compute_diversity(
        data_iterator=iter(all_records),
        dataset_name=dataset_name,
        output_file=None,
        **kwargs,
    )

    base_results['summary']['total_sessions'] = len(sessions)
    base_results['summary']['has_multi_turn'] = has_multi_turn

    if has_multi_turn:
        session_div = _compute_session_diversity(sessions)
        base_results['dimensions']['session_diversity'] = session_div

    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(base_results, f, indent=2, ensure_ascii=False)
        print(f"saved: {output_file}")

    return base_results


# ============================================================================
# session-level 多样性
# ============================================================================

def _compute_session_diversity(sessions: List[Session]) -> Dict[str, Any]:
    """
    计算 session-level 多样性

    1. 轮次数分布多样性
    2. 轮间语义转移多样性
    3. 轮间长度模式多样性
    """
    results: Dict[str, Any] = {}

    # --- 1. 轮次数分布多样性 ---
    round_counts = Counter(len(s.rounds) for s in sessions)
    round_count_entropy, _, n_unique_counts = _compute_entropy(round_counts)
    results['round_count_entropy'] = round_count_entropy
    results['n_unique_round_counts'] = n_unique_counts
    results['round_count_distribution'] = dict(round_counts.most_common())

    # --- 2. 轮间语义转移多样性 ---
    transition_pairs = []
    for session in sessions:
        for i in range(len(session.rounds) - 1):
            instr_curr = session.rounds[i].instruction or ""
            instr_next = session.rounds[i + 1].instruction or ""
            if instr_curr and instr_next:
                transition_pairs.append((instr_curr, instr_next))

    if len(transition_pairs) >= 2:
        try:
            from sentence_transformers import SentenceTransformer

            local_model_path = os.path.join(_st_dir, 'models', 'all-MiniLM-L6-v2')
            if os.path.exists(local_model_path):
                model = SentenceTransformer(local_model_path)
            else:
                model = SentenceTransformer('all-MiniLM-L6-v2')

            curr_emb = model.encode(
                [p[0] for p in transition_pairs],
                convert_to_numpy=True, normalize_embeddings=True,
            )
            next_emb = model.encode(
                [p[1] for p in transition_pairs],
                convert_to_numpy=True, normalize_embeddings=True,
            )

            transition_vectors = next_emb - curr_emb
            norms = np.linalg.norm(transition_vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized = transition_vectors / norms
            sim_matrix = np.dot(normalized, normalized.T)
            n = len(transition_pairs)
            upper_tri = sim_matrix[np.triu_indices(n, k=1)]

            results['transition_diversity'] = 1.0 - float(np.mean(upper_tri))
            results['avg_transition_similarity'] = float(np.mean(upper_tri))
            results['n_transitions'] = n
        except ImportError:
            results['transition_diversity'] = None
            results['n_transitions'] = len(transition_pairs)
    else:
        results['transition_diversity'] = None
        results['n_transitions'] = len(transition_pairs)

    # --- 3. 轮间长度模式多样性 ---
    length_patterns = []
    for session in sessions:
        if session.is_multi_turn:
            pattern = tuple(len(r.actions) for r in session.rounds)
            length_patterns.append(pattern)

    if length_patterns:
        pattern_counter = Counter(length_patterns)
        pattern_entropy, _, n_unique_patterns = _compute_entropy(pattern_counter)
        results['length_pattern_entropy'] = pattern_entropy
        results['n_unique_length_patterns'] = n_unique_patterns
        results['unique_length_pattern_ratio'] = n_unique_patterns / len(length_patterns)
        results['top_length_patterns'] = {
            str(k): v for k, v in pattern_counter.most_common(10)
        }
    else:
        results['length_pattern_entropy'] = 0.0
        results['n_unique_length_patterns'] = 0
        results['unique_length_pattern_ratio'] = 0.0

    return results
