#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution Similarity 分布相似度 (基于 MMD)

衡量"生成数据分布"与"真实数据分布"的贴合程度。
对 query, tools, api_calls 三个字段分别计算 MMD 距离。

使用方式:
    from distribution_similarity import compute_distribution_similarity
    from loaders import ToolBenchLoader, ArceeAgentDataLoader

    results = compute_distribution_similarity(
        real_iterator=ToolBenchLoader('/path/to/real.json').iterate(),
        gen_iterator=ArceeAgentDataLoader('/path/to/gen.jsonl').iterate(),
        real_name='ToolBench',
        gen_name='Arcee',
    )

底层也可直接使用 compute_mmd() 传入已有的 embedding numpy 数组。
"""

import json
import os
import time
from datetime import datetime
from typing import Sequence, Dict, Any, List, Optional, Iterator

import numpy as np
from tqdm import tqdm

import sys
_metrics_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_api_agent_dir = os.path.dirname(_metrics_dir)
sys.path.insert(0, _api_agent_dir)
sys.path.insert(0, _metrics_dir)

from data_types import APIAgentSample  # noqa: E402

FIELDS = ["query", "tools", "api_calls"]


# =============================================================================
# 底层 MMD 计算
# =============================================================================

def _rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    """RBF 核: k(x, y) = exp(- ||x - y||^2 / (2 * sigma^2))"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    sq_dists = x_norm + y_norm - 2.0 * (x @ y.T)
    sigma2 = 2.0 * (bandwidth ** 2)
    return np.exp(-sq_dists / sigma2)


def compute_mmd(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    kernel: str = "rbf",
    bandwidths: Sequence[float] = (1.0,),
) -> Dict[str, Any]:
    """
    计算两组特征分布之间的 MMD^2。

    Args:
        real_features: 真实数据特征 (n_r, d)
        gen_features: 生成数据特征 (n_g, d)
        kernel: 核类型，目前仅支持 "rbf"
        bandwidths: RBF 核带宽列表，多带宽求平均

    Returns:
        {"mmd2": float, "similarity_score": float, ...}
    """
    real = np.asarray(real_features, dtype=float)
    gen = np.asarray(gen_features, dtype=float)

    if real.ndim != 2 or gen.ndim != 2:
        raise ValueError("real_features 和 gen_features 必须是二维矩阵 (n, d)")
    if real.shape[1] != gen.shape[1]:
        raise ValueError(f"特征维度不匹配：real {real.shape} vs gen {gen.shape}")
    if kernel != "rbf":
        raise ValueError(f"目前仅支持 kernel='rbf'，收到: {kernel}")
    if not bandwidths:
        raise ValueError("bandwidths 不能为空")

    n_r, n_g = real.shape[0], gen.shape[0]
    mmd2_values: List[float] = []
    for bw in bandwidths:
        k_xx = _rbf_kernel(real, real, bw)
        k_yy = _rbf_kernel(gen, gen, bw)
        k_xy = _rbf_kernel(real, gen, bw)
        mmd2 = float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))
        mmd2_values.append(mmd2)

    mmd2_avg = float(np.mean(mmd2_values))
    return {
        "mmd2": mmd2_avg,
        "similarity_score": 1.0 / (1.0 + max(mmd2_avg, 0.0)),
        "kernel": kernel,
        "bandwidths": list(bandwidths),
        "n_real": int(n_r),
        "n_gen": int(n_g),
        "per_bandwidth_mmd2": mmd2_values,
    }


# =============================================================================
# 高层接口
# =============================================================================

def compute_distribution_similarity(
    real_iterator: Iterator[APIAgentSample],
    gen_iterator: Iterator[APIAgentSample],
    real_name: str = "Real",
    gen_name: str = "Generated",
    embedding_model: str = "all-MiniLM-L6-v2",
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0),
    max_samples: Optional[int] = None,
    embedding_batch_size: int = 64,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    计算两个数据集在 query/tools/api_calls 三个字段上的分布相似度。

    对每个字段分别生成 embedding，然后用 MMD 计算分布距离。

    Returns:
        {
            "per_field": {
                "query": {"mmd2": float, "similarity_score": float, ...},
                "tools": {...},
                "api_calls": {...},
            },
            ...
        }
    """
    from diversity import generate_embeddings

    print("=" * 70)
    print("Distribution Similarity 分布相似度评估")
    print("=" * 70)
    print(f"真实数据集: {real_name} | 生成数据集: {gen_name}")
    print(f"字段: {', '.join(FIELDS)}")
    print(f"模型: {embedding_model}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    # 收集样本（需要多次遍历，所以先 materialize）
    print("-" * 50)
    print("收集样本")
    print("-" * 50)
    real_list: List[APIAgentSample] = []
    for s in tqdm(real_iterator, desc=f"收集 {real_name}"):
        if max_samples is not None and len(real_list) >= max_samples:
            break
        real_list.append(s)

    gen_list: List[APIAgentSample] = []
    for s in tqdm(gen_iterator, desc=f"收集 {gen_name}"):
        if max_samples is not None and len(gen_list) >= max_samples:
            break
        gen_list.append(s)

    print(f"  {real_name}: {len(real_list):,} 样本")
    print(f"  {gen_name}: {len(gen_list):,} 样本")
    print()

    # 对每个字段计算 MMD
    per_field = {}
    for f in FIELDS:
        print("-" * 50)
        print(f"字段: {f}")
        print("-" * 50)

        real_emb = generate_embeddings(
            data_iterator=iter(real_list), field=f,
            model_name=embedding_model, batch_size=embedding_batch_size,
        )
        gen_emb = generate_embeddings(
            data_iterator=iter(gen_list), field=f,
            model_name=embedding_model, batch_size=embedding_batch_size,
        )

        mmd_result = compute_mmd(
            real_features=real_emb, gen_features=gen_emb,
            kernel="rbf", bandwidths=bandwidths,
        )
        per_field[f] = mmd_result
        print(f"  MMD²: {mmd_result['mmd2']:.6f}  Similarity: {mmd_result['similarity_score']:.6f}")
        print()

    total_time = time.time() - start_time

    results = {
        "real_dataset": real_name,
        "gen_dataset": gen_name,
        "embedding_model": embedding_model,
        "fields": FIELDS,
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "n_real": len(real_list),
        "n_gen": len(gen_list),
        "per_field": per_field,
    }

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_file}")

    print()
    print("=" * 70)
    print(f"评估完成！总耗时 {total_time:.1f} 秒")
    print("=" * 70)
    print(f"  {real_name} ({len(real_list):,}) vs {gen_name} ({len(gen_list):,})")
    for f in FIELDS:
        mmd2 = per_field[f]["mmd2"]
        sim = per_field[f]["similarity_score"]
        print(f"  {f:12s}  MMD²: {mmd2:.6f}  Similarity: {sim:.6f}")
    print()

    return results


__all__ = [
    "compute_mmd",
    "compute_distribution_similarity",
]
