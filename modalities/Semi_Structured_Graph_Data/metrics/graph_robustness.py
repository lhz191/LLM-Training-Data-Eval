"""
Graph Robustness — σSCR / GAD / GADcap / DL2  (Survey §5.3.1)

指标来源: Richardeau (Le Merrer & Trédan), 2025
         "LLMs hallucinate graphs too: a structural perspective"

1. Syntactic Correctness Rate (σSCR)
   生成输出能被解析为合法图结构的比率。

2. Graph Atlas Distance (GAD / GADcap)
   生成图与目标 canonical atlas 图之间的平均图编辑距离 (GED)。
   GADcap 对极端值截断以增强鲁棒性。

3. Degree-distribution Deviation (DL2)
   生成图与参考图归一化度分布直方图的 L2 距离。

依赖: networkx, numpy
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import networkx as nx


# ═══════════════════════════════════════════════════════════
#  1. Syntactic Correctness Rate (σSCR)
# ═══════════════════════════════════════════════════════════

def compute_scr(
    raw_outputs: List[str],
    parse_fn: Callable[[str], Optional[nx.Graph]],
) -> Dict[str, Any]:
    """
    σSCR = (1/N) Σ 1{ParseOK(G)}

    参数:
        raw_outputs: LLM 的原始文本输出列表
        parse_fn:    解析函数，接受字符串，返回 nx.Graph 或 None（解析失败）
    """
    n = len(raw_outputs)
    parsed = [parse_fn(s) for s in raw_outputs]
    ok = sum(1 for g in parsed if g is not None)
    return {
        'scr': ok / n if n else 0.0,
        'num_parsed': ok,
        'num_total': n,
    }


# ═══════════════════════════════════════════════════════════
#  2. Graph Atlas Distance (GAD / GADcap)
# ═══════════════════════════════════════════════════════════

def compute_gad(
    generated_graphs: List[nx.Graph],
    target_graphs: List[nx.Graph],
    cap: Optional[float] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    GAD     = (1/k) Σ d_GED(Gi, ai)
    GADcap  = (1/k) Σ min(d_GED(Gi, ai), C)

    参数:
        generated_graphs: 生成图列表
        target_graphs:    对应的目标图列表 (如 atlas canonical graphs)
        cap:              GADcap 的截断值 C，None 则不截断
        timeout:          单次 GED 计算超时 (秒)，超时抛出 TimeoutError
    """
    assert len(generated_graphs) == len(target_graphs), "列表长度不一致"
    k = len(generated_graphs)

    dists = []
    for g_gen, g_tgt in zip(generated_graphs, target_graphs):
        ged = nx.graph_edit_distance(g_gen, g_tgt, timeout=timeout)
        if ged is None:
            raise TimeoutError(
                f"GED 计算超时 ({timeout}s)，图规模 "
                f"({g_gen.number_of_nodes()}, {g_tgt.number_of_nodes()} nodes)。"
                f"请增大 timeout 或使用更小的图。"
            )
        dists.append(float(ged))

    dists_arr = np.array(dists)
    gad = float(np.mean(dists_arr))

    results: Dict[str, Any] = {
        'gad': gad,
        'num_pairs': k,
    }

    if cap is not None:
        capped = np.minimum(dists_arr, cap)
        results['gad_capped'] = float(np.mean(capped))
        results['cap_value'] = cap

    return results


# ═══════════════════════════════════════════════════════════
#  3. Degree-distribution Deviation (DL2)
# ═══════════════════════════════════════════════════════════

def _normalized_degree_hist(G: nx.Graph) -> np.ndarray:
    """返回归一化度分布直方图 (概率向量)。"""
    h = nx.degree_histogram(G)
    h = np.array(h, dtype=np.float64)
    total = h.sum()
    if total > 0:
        h /= total
    return h


def _pad_to_same_length(h1: np.ndarray, h2: np.ndarray) -> tuple:
    """将两个直方图零填充到相同长度。"""
    max_len = max(len(h1), len(h2))
    h1_pad = np.zeros(max_len)
    h2_pad = np.zeros(max_len)
    h1_pad[:len(h1)] = h1
    h2_pad[:len(h2)] = h2
    return h1_pad, h2_pad


def compute_dl2(
    generated_graph: nx.Graph,
    reference_graph: nx.Graph,
) -> float:
    """
    DL2(G) = || h_gen(G) - h_ref ||_2

    参数:
        generated_graph: 生成图
        reference_graph: 参考图 (ground truth 或 reference dataset 的代表)

    返回:
        L2 距离 (float)
    """
    h_gen = _normalized_degree_hist(generated_graph)
    h_ref = _normalized_degree_hist(reference_graph)
    h_gen, h_ref = _pad_to_same_length(h_gen, h_ref)
    return float(np.linalg.norm(h_gen - h_ref, ord=2))


def compute_dl2_batch(
    generated_graphs: List[nx.Graph],
    reference_graphs: List[nx.Graph],
) -> Dict[str, Any]:
    """
    批量计算 DL2，返回均值和逐对结果。

    参数:
        generated_graphs: 生成图列表
        reference_graphs: 对应参考图列表 (一对一)
    """
    assert len(generated_graphs) == len(reference_graphs)
    scores = [compute_dl2(g, r) for g, r in zip(generated_graphs, reference_graphs)]
    return {
        'dl2_mean': float(np.mean(scores)),
        'dl2_std': float(np.std(scores)),
        'dl2_scores': scores,
        'num_pairs': len(scores),
    }


# ═══════════════════════════════════════════════════════════
#  聚合入口
# ═══════════════════════════════════════════════════════════

def compute_graph_robustness(
    raw_outputs: List[str] = None,
    parse_fn: Callable[[str], Optional[nx.Graph]] = None,
    generated_graphs: List[nx.Graph] = None,
    target_graphs: List[nx.Graph] = None,
    reference_graphs: List[nx.Graph] = None,
    cap: Optional[float] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    一站式计算所有 Graph Robustness 指标。

    参数:
        raw_outputs + parse_fn:                → σSCR
        generated_graphs + target_graphs:      → GAD / GADcap
        generated_graphs + reference_graphs:   → DL2
        cap:    GADcap 截断值
        timeout: GED 超时秒数
    """
    results: Dict[str, Any] = {}

    if raw_outputs is not None and parse_fn is not None:
        results.update(compute_scr(raw_outputs, parse_fn))

    if generated_graphs is not None and target_graphs is not None:
        results.update(compute_gad(generated_graphs, target_graphs, cap, timeout))

    if generated_graphs is not None and reference_graphs is not None:
        results.update(compute_dl2_batch(generated_graphs, reference_graphs))

    return results
