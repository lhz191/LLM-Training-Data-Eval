#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Turn API Agent Diversity

在单轮基础上，增加 session-level 多样性分析。

=== 设计 ===
1. 提取所有 APIAgentSample，调用单轮 compute_diversity 获取基础指标
2. 如果存在多轮 session (rounds > 1)，额外计算 session-level 指标：
   - 轮次数分布多样性
   - 轮间查询语义转移多样性
   - 轮间工具使用模式多样性
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
_st_dir = os.path.join(_agent_data_dir, 'api_agent_eval')

# ---------------------------------------------------------------------------
# 从多轮 data_types 导入 Session / APIAgentSample
# ---------------------------------------------------------------------------
if _mt_dir not in sys.path:
    sys.path.insert(0, _mt_dir)
from data_types import Session, APIAgentSample  # noqa: E402

# ---------------------------------------------------------------------------
# 显式加载单轮 diversity 模块（避免和本文件同名冲突）
# ---------------------------------------------------------------------------
_st_div_path = os.path.join(_st_dir, 'metrics', 'diversity.py')
_spec = importlib.util.spec_from_file_location('st_api_diversity', _st_div_path)
_st_diversity = importlib.util.module_from_spec(_spec)

if _st_dir not in sys.path:
    sys.path.append(_st_dir)
_spec.loader.exec_module(_st_diversity)

compute_diversity = _st_diversity.compute_diversity


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
    计算多轮 API Agent 数据集的多样性指标

    内部提取所有 APIAgentSample 调用单轮 compute_diversity，
    然后追加 session-level 多样性分析。
    """
    sessions: List[Session] = []
    all_samples: List[APIAgentSample] = []
    for session in session_iterator:
        if max_sessions and len(sessions) >= max_sessions:
            break
        sessions.append(session)
        all_samples.extend(session.rounds)

    has_multi_turn = any(s.is_multi_turn for s in sessions)

    base_results = compute_diversity(
        data_iterator=iter(all_samples),
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

    return base_results


# ============================================================================
# session-level 多样性
# ============================================================================

def _compute_session_diversity(sessions: List[Session]) -> Dict[str, Any]:
    """
    session-level 多样性

    1. 轮次数分布多样性
    2. 轮间查询语义转移多样性
    3. 轮间工具使用模式多样性
    """
    results: Dict[str, Any] = {}

    # --- 1. 轮次数分布 ---
    round_counts = Counter(len(s.rounds) for s in sessions)
    results['round_count_distribution'] = dict(round_counts.most_common())
    results['n_unique_round_counts'] = len(round_counts)

    # --- 2. 轮间查询语义转移多样性 ---
    transition_pairs = []
    for session in sessions:
        for i in range(len(session.rounds) - 1):
            q_curr = session.rounds[i].query or ""
            q_next = session.rounds[i + 1].query or ""
            if q_curr and q_next:
                transition_pairs.append((q_curr, q_next))

    if len(transition_pairs) >= 2:
        try:
            from sentence_transformers import SentenceTransformer

            local_model_path = os.path.join(_st_dir, 'metrics', 'models', 'all-MiniLM-L6-v2')
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
            results['n_transitions'] = n
        except ImportError:
            results['transition_diversity'] = None
            results['n_transitions'] = len(transition_pairs)
    else:
        results['transition_diversity'] = None
        results['n_transitions'] = len(transition_pairs)

    # --- 3. 轮间工具使用模式多样性 ---
    tool_patterns = []
    for session in sessions:
        if session.is_multi_turn:
            pattern = tuple(
                tuple(sorted(s.get_tool_names())) for s in session.rounds
            )
            tool_patterns.append(pattern)

    if tool_patterns:
        unique_patterns = len(set(tool_patterns))
        results['n_unique_tool_patterns'] = unique_patterns
        results['unique_tool_pattern_ratio'] = unique_patterns / len(tool_patterns)
    else:
        results['n_unique_tool_patterns'] = 0
        results['unique_tool_pattern_ratio'] = 0.0

    return results
