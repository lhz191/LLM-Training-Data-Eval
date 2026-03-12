"""
Graph Diversity — Novelty & Uniqueness  (Survey §5.2.1, Yao et al. 2024)

Novelty (novel rate):
    Novelty = (1/|G_gen|) Σ_{G∈G_gen} I{G ≢ G_ex}
    G_ex 是 prompt 中的示例图，G ≢ G_ex 表示 G 与所有示例图都不同构。

Uniqueness (unique rate):
    Uniq = |unique(G_valid)| / |G_valid|
    G_valid = {G ∈ G_gen : passes_rules(G)}，即先筛有效图再去重。
"""

from typing import List, Dict, Any

import networkx as nx


def compute_graph_diversity(
    syn_graphs: List,
    example_graphs: List = None,
    valid_mask: List[bool] = None,
    max_compare: int = 500,
) -> Dict[str, Any]:
    """
    计算生成图的 Novelty 和 Uniqueness。

    参数:
        syn_graphs:      生成图列表 (nx.Graph)
        example_graphs:  prompt 中的示例图列表 (用于 Novelty)
        valid_mask:      与 syn_graphs 等长的布尔列表，True 表示该图通过了 validity 检查。
                         如果为 None，则所有图都视为 valid。
        max_compare:     最大比较数（避免 O(n²) 爆炸）

    返回:
        dict: novelty_rate, uniqueness, num_unique, num_valid, ...
    """
    n = len(syn_graphs)
    if n == 0:
        return {'uniqueness': 0.0, 'novelty_rate': 0.0, 'num_syn': 0}

    # ── Uniqueness: 在 valid 图上去重 (Survey 定义) ────────
    if valid_mask is not None:
        valid_graphs = [g for g, v in zip(syn_graphs, valid_mask) if v]
    else:
        valid_graphs = syn_graphs

    nv = len(valid_graphs)
    cap = min(nv, max_compare)

    unique_count = 0
    is_dup = [False] * cap
    for i in range(cap):
        if is_dup[i]:
            continue
        unique_count += 1
        for j in range(i + 1, cap):
            if is_dup[j]:
                continue
            if nx.is_isomorphic(valid_graphs[i], valid_graphs[j]):
                is_dup[j] = True

    uniqueness = unique_count / cap if cap > 0 else 0.0

    # ── Novelty: 与示例图都不同构的比例 ───────────────────
    results = {
        'uniqueness': float(uniqueness),
        'num_syn': n,
        'num_valid': nv,
        'num_unique': unique_count,
    }

    if example_graphs and len(example_graphs) > 0:
        cap_n = min(n, max_compare)
        novel_count = 0

        for i in range(cap_n):
            is_novel = True
            for ex in example_graphs:
                if nx.is_isomorphic(syn_graphs[i], ex):
                    is_novel = False
                    break
            if is_novel:
                novel_count += 1

        results['novelty_rate'] = novel_count / cap_n if cap_n > 0 else 0.0
        results['num_examples'] = len(example_graphs)

    return results
