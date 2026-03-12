"""
来自 LLM4GraphGen (Yao et al., 2024) 的 8 种图生成规则检查器。

论文: "Exploring the Potential of Large Language Models in Graph Generation"
arXiv: 2403.14358

【重要】本文件中的 8 条 rule 均针对无向图定义。
如果传入有向图 (DiGraph)，规则会直接返回 False 并提示类型不匹配。
有向图的规则检查器（如 DAG、强连通、arborescence 等）应另建文件。

每个 rule checker 的签名统一为:
    def rule_xxx(G: nx.Graph, spec: dict) -> (bool, str)
        G:    networkx 图对象（应为无向图）
        spec: 任务参数字典，如 {"num_nodes": 15, "k": 3}
        返回: (是否通过, 说明文字)

所有 rule 内部会自动检查 spec 中的 num_nodes / num_edges（如果提供了的话），
与论文实验设置一致（每个任务都指定了节点数，部分还指定了边数）。
"""

import networkx as nx
from typing import Dict, Tuple, Optional


def _check_undirected(G: nx.Graph) -> Optional[str]:
    """本文件所有 rule 仅适用于无向图，有向图直接拒绝。"""
    if G.is_directed():
        return "本规则仅适用于无向图，传入了有向图 (DiGraph)"
    return None


def _check_size(G: nx.Graph, spec: Dict) -> Optional[str]:
    """
    通用的节点数/边数检查。
    如果 spec 中有 num_nodes 或 num_edges，检查是否匹配。
    返回 None 表示通过，否则返回失败原因字符串。
    """
    expected_n = spec.get("num_nodes")
    if expected_n is not None and G.number_of_nodes() != expected_n:
        return f"|V|={G.number_of_nodes()}, 期望{expected_n}"

    expected_m = spec.get("num_edges")
    if expected_m is not None and G.number_of_edges() != expected_m:
        return f"|E|={G.number_of_edges()}, 期望{expected_m}"

    return None


# ── 1. Tree ──────────────────────────────────────────────
# 论文: "a tree with the specified number of nodes" (Table 7: 15 nodes)
def rule_tree(G: nx.Graph, spec: Dict) -> Tuple[bool, str]:
    if G.number_of_nodes() == 0:
        return False, "空图"
    dir_err = _check_undirected(G)
    if dir_err:
        return False, dir_err
    size_err = _check_size(G, spec)
    if size_err:
        return False, size_err
    ok = nx.is_tree(G)
    return ok, "是树" if ok else "不是树"


# ── 2. Cycle ─────────────────────────────────────────────
# 论文: "a cycle with the specified number of nodes, no other nodes or edges"
def rule_cycle(G: nx.Graph, spec: Dict) -> Tuple[bool, str]:
    dir_err = _check_undirected(G)
    if dir_err:
        return False, dir_err
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        return False, "空图"
    size_err = _check_size(G, spec)
    if size_err:
        return False, size_err
    if not nx.is_connected(G):
        return False, "不连通"
    if m != n:
        return False, f"|E|={m} != |V|={n}"
    if any(d != 2 for _, d in G.degree()):
        return False, "存在度数≠2的节点"
    return True, "是环"


# ── 3. Planar ────────────────────────────────────────────
# 论文: "a planar graph with the specified number of nodes and edges"
# (Table 7: 15 nodes, 24 edges)
def rule_planar(G: nx.Graph, spec: Dict) -> Tuple[bool, str]:
    dir_err = _check_undirected(G)
    if dir_err:
        return False, dir_err
    size_err = _check_size(G, spec)
    if size_err:
        return False, size_err
    ok, _ = nx.check_planarity(G)
    return ok, "平面图" if ok else "非平面图"


# ── 4. Components ────────────────────────────────────────
# 论文: "a specified number of connected components"
# (Table 7: 15 nodes, 5 components)
def rule_components(G: nx.Graph, spec: Dict) -> Tuple[bool, str]:
    dir_err = _check_undirected(G)
    if dir_err:
        return False, dir_err
    size_err = _check_size(G, spec)
    if size_err:
        return False, size_err
    expected = spec.get("num_components")
    if expected is None:
        return True, "未指定分量数约束"
    actual = nx.number_connected_components(G)
    ok = actual == expected
    return ok, f"连通分量={actual}" + ("" if ok else f" (期望{expected})")


# ── 5. k-Regular ─────────────────────────────────────────
# 论文: "the degree of every node is k"
# (Table 7: 16 nodes, k=3)
def rule_k_regular(G: nx.Graph, spec: Dict) -> Tuple[bool, str]:
    dir_err = _check_undirected(G)
    if dir_err:
        return False, dir_err
    size_err = _check_size(G, spec)
    if size_err:
        return False, size_err
    k = spec.get("k")
    if k is None:
        return True, "未指定 k"
    degs = [d for _, d in G.degree()]
    if not degs:
        return False, "空图"
    ok = all(d == k for d in degs)
    return ok, f"{k}-正则图" if ok else f"度数分布{set(degs)}, 不是{k}-正则"


# ── 6. Wheel ─────────────────────────────────────────────
# 论文: "connecting a single node to all nodes of a cycle"
# (Table 7: 15 nodes)
def rule_wheel(G: nx.Graph, spec: Dict) -> Tuple[bool, str]:
    dir_err = _check_undirected(G)
    if dir_err:
        return False, dir_err
    n = G.number_of_nodes()
    if n < 4:
        return False, f"节点数{n}<4，不可能是轮图"
    size_err = _check_size(G, spec)
    if size_err:
        return False, size_err
    if not nx.is_connected(G):
        return False, "不连通"
    degs = dict(G.degree())
    hubs = [v for v, d in degs.items() if d == n - 1]
    if len(hubs) != 1:
        return False, f"hub数量={len(hubs)}，期望1"
    hub = hubs[0]
    rim = [v for v in G.nodes() if v != hub]
    if any(degs[v] != 3 for v in rim):
        return False, "非hub节点度数不全为3"
    rim_subgraph = G.subgraph(rim)
    if not (nx.is_connected(rim_subgraph) and
            rim_subgraph.number_of_edges() == len(rim) and
            all(d == 2 for _, d in rim_subgraph.degree())):
        return False, "外圈不构成环"
    return True, "是轮图"


# ── 7. Bipartite ─────────────────────────────────────────
# 论文: "two disjoint sets U and V with specified sizes"
# (Table 7: 5 nodes in each partition)
def rule_bipartite(G: nx.Graph, spec: Dict) -> Tuple[bool, str]:
    dir_err = _check_undirected(G)
    if dir_err:
        return False, dir_err
    size_err = _check_size(G, spec)
    if size_err:
        return False, size_err
    ok = nx.is_bipartite(G)
    if not ok:
        return False, "不是二部图"
    sizes = spec.get("partition_sizes")  # 如 [5, 5]
    if sizes:
        parts = nx.bipartite.sets(G)
        actual = sorted([len(parts[0]), len(parts[1])])
        expected = sorted(sizes)
        if actual != expected:
            return False, f"分区大小{actual} != 期望{expected}"
    return True, "二部图"


# ── 8. k-Colorable ───────────────────────────────────────
# 论文: "k-colorable, each node is assigned a color, adjacent nodes differ"
# (Table 7: 15 nodes, 32 edges, k=3)
def rule_k_colorable(G: nx.Graph, spec: Dict) -> Tuple[bool, str]:
    dir_err = _check_undirected(G)
    if dir_err:
        return False, dir_err
    size_err = _check_size(G, spec)
    if size_err:
        return False, size_err
    k = spec.get("k")
    if k is None:
        return True, "未指定 k"

    # 先用贪心快速判断：贪心 ≤ k 则一定可着色，直接返回
    greedy = nx.coloring.greedy_color(G, strategy="DSATUR")
    greedy_k = len(set(greedy.values())) if greedy else 0
    if greedy_k <= k:
        return True, f"色数≤{greedy_k} ≤ {k}"

    # 贪心失败不代表不可着色，用回溯精确求解
    # 论文中图规模小 (≤50 nodes)，回溯可行
    if _exact_k_colorable(G, k):
        return True, f"精确求解: {k}-可着色"
    return False, f"精确求解: 不是{k}-可着色"


def _exact_k_colorable(G: nx.Graph, k: int) -> bool:
    """回溯法精确判定图 G 是否 k-可着色。适用于小图 (≤~50 nodes)。"""
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return True

    node_idx = {v: i for i, v in enumerate(nodes)}
    adj = [[] for _ in range(n)]
    for u, v in G.edges():
        adj[node_idx[u]].append(node_idx[v])
        adj[node_idx[v]].append(node_idx[u])

    color = [0] * n

    def backtrack(i: int) -> bool:
        if i == n:
            return True
        neighbor_colors = {color[nb] for nb in adj[i] if color[nb] != 0}
        for c in range(1, k + 1):
            if c not in neighbor_colors:
                color[i] = c
                if backtrack(i + 1):
                    return True
                color[i] = 0
        return False

    return backtrack(0)


# ── 所有 rules 注册表 ────────────────────────────────────
RULES = {
    "tree": rule_tree,
    "cycle": rule_cycle,
    "planar": rule_planar,
    "components": rule_components,
    "k_regular": rule_k_regular,
    "wheel": rule_wheel,
    "bipartite": rule_bipartite,
    "k_colorable": rule_k_colorable,
}
