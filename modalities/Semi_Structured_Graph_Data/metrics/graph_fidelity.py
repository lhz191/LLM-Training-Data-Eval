"""
Graph Fidelity — MMD + FCD  (Survey §5.2.1)

使用 GGM-metrics (O'Bray et al.) 的实现计算图分布间的 MMD:
  - Degree MMD (GraphRNN, You et al. 2018)
  - Clustering MMD
  - Orbits MMD (需要编译 orca 二进制)
  - Spectral MMD

分子图额外计算 FCD (Fréchet ChemNet Distance, Preuer et al. 2018)。

注意: GGM-metrics 还提供 WL MMD 和 NSPDK MMD，但它们的 evaluate() 接口
假定输入为 DGL 图，不能直接传 nx.Graph。如需使用，需先将 nx.Graph 转为 DGL 格式。

依赖: pyemd, dgl, networkx, scipy, scikit-learn, fcd (分子图)
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional

_GGM_EVAL_DIR = os.path.join(os.path.dirname(__file__), 'GGM-metrics', 'evaluation')
if _GGM_EVAL_DIR not in sys.path:
    sys.path.insert(0, _GGM_EVAL_DIR)

from graph_structure_evaluation import MMDEval

try:
    from fcd import get_fcd
    HAS_FCD = True
except ImportError:
    HAS_FCD = False


# ════════════════════════════════════════════════════════════
# 结构 MMD (degree / clustering / orbits / spectral)
# ════════════════════════════════════════════════════════════

# GGM-metrics 中 Orbits 的 orca 路径是相对路径，需要 cd 到 GGM-metrics 目录
_GGM_ROOT = os.path.join(os.path.dirname(__file__), 'GGM-metrics')


def compute_graph_mmd(
    syn_graphs: List,
    real_graphs: List,
    statistics: List[str] = None,
    is_parallel: bool = False,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """
    计算生成图与真实图之间的 MMD。

    参数:
        syn_graphs:   生成图列表 (nx.Graph)
        real_graphs:  真实图列表 (nx.Graph)
        statistics:   要计算的描述符列表，可选:
                      "degree", "clustering", "orbits", "spectral"
                      默认 ["degree", "clustering", "orbits", "spectral"]
        is_parallel:  是否并行提取特征
        max_workers:  并行 worker 数

    返回:
        dict: 每种统计量的 MMD² 值 + 平均值
    """
    if statistics is None:
        statistics = ["degree", "clustering", "orbits", "spectral"]

    results = {}
    saved_cwd = os.getcwd()

    try:
        # orbits 需要在 GGM-metrics 目录下运行（orca 路径是相对的）
        os.chdir(_GGM_ROOT)

        for stat in statistics:
            evaluator = MMDEval(
                statistic=stat,
                is_parallel=is_parallel,
                max_workers=max_workers)
            res, t = evaluator.evaluate(
                generated_dataset=syn_graphs,
                reference_dataset=real_graphs)
            results.update(res)
            results[f'{stat}_mmd_time'] = t

    finally:
        os.chdir(saved_cwd)

    mmd_values = [v for k, v in results.items()
                  if k.endswith('_mmd') and not k.endswith('_time')]
    if mmd_values:
        results['mmd_avg'] = float(np.mean(mmd_values))

    results['num_real'] = len(real_graphs)
    results['num_syn'] = len(syn_graphs)

    return results


# ════════════════════════════════════════════════════════════
# FCD — 分子图专用 (Preuer et al. 2018)
# ════════════════════════════════════════════════════════════

def compute_fcd(
    syn_smiles: List[str],
    real_smiles: List[str],
) -> Dict[str, Any]:
    """
    计算分子图的 Fréchet ChemNet Distance。

    参数:
        syn_smiles:  生成分子的 SMILES 列表
        real_smiles: 真实分子的 SMILES 列表

    返回:
        dict: {"fcd": float}
    """
    if not HAS_FCD:
        return {'error': 'fcd 未安装 (pip install fcd)'}

    fcd_score = get_fcd(real_smiles, syn_smiles)
    return {
        'fcd': float(fcd_score),
        'num_real': len(real_smiles),
        'num_syn': len(syn_smiles),
    }
