#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution Similarity 分布相似度 (基于 MMD)

衡量"生成数据分布"与"真实数据分布"的贴合程度。
内部自动生成 embedding，用户只需传入两个数据集的 iterator。

使用方式:
    from distribution_similarity import compute_distribution_similarity
    from loaders import ToolBenchLoader, ArceeAgentDataLoader

    results = compute_distribution_similarity(
        real_iterator=ToolBenchLoader('/path/to/real.json').iterate(),
        gen_iterator=ArceeAgentDataLoader('/path/to/gen.jsonl').iterate(),
        real_name='ToolBench',
        gen_name='Arcee',
        field='query',
        output_file='results/distribution_similarity.json',
    )

底层也可直接使用 compute_mmd() 传入已有的 embedding numpy 数组。
"""

import json
import os
import time
from datetime import datetime
from typing import Sequence, Dict, Any, List, Optional, Iterator

import numpy as np

import sys
_metrics_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_api_agent_dir = os.path.dirname(_metrics_dir)
sys.path.insert(0, _api_agent_dir)
sys.path.insert(0, _metrics_dir)

from data_types import APIAgentSample  # noqa: E402


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

    n_r = real.shape[0]
    n_g = gen.shape[0]

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
# 高层接口：传入 iterator，自动生成 embedding 并计算
# =============================================================================

def compute_distribution_similarity(
    real_iterator: Iterator[APIAgentSample],
    gen_iterator: Iterator[APIAgentSample],
    real_name: str = "Real",
    gen_name: str = "Generated",
    field: str = "query",
    embedding_model: str = "all-MiniLM-L6-v2",
    bandwidths: Sequence[float] = (0.5, 1.0, 2.0),
    max_samples: Optional[int] = None,
    real_cache_path: Optional[str] = None,
    gen_cache_path: Optional[str] = None,
    embedding_batch_size: int = 64,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    计算两个数据集的分布相似度。

    自动为两侧数据生成 embedding，然后用 MMD 计算分布距离。

    Args:
        real_iterator: 真实数据集的 APIAgentSample 迭代器
        gen_iterator: 生成数据集的 APIAgentSample 迭代器
        real_name: 真实数据集名称
        gen_name: 生成数据集名称
        field: embedding 字段 ('query', 'tools', 'both')
        embedding_model: embedding 模型名称
        bandwidths: RBF 核多尺度带宽
        max_samples: 每侧最大样本数
        real_cache_path: 真实数据 embedding 缓存路径
        gen_cache_path: 生成数据 embedding 缓存路径
        embedding_batch_size: embedding batch 大小
        output_file: 结果保存路径

    Returns:
        包含 MMD²、similarity_score 等指标的字典
    """
    from diversity import generate_embeddings

    print("=" * 70)
    print("Distribution Similarity 分布相似度评估")
    print("=" * 70)
    print(f"真实数据集: {real_name}")
    print(f"生成数据集: {gen_name}")
    print(f"字段: {field}")
    print(f"模型: {embedding_model}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    # Step 1: 生成真实数据 embedding
    print("-" * 50)
    print(f"Step 1: 生成 {real_name} embedding")
    print("-" * 50)
    real_emb = generate_embeddings(
        data_iterator=real_iterator,
        field=field,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        max_samples=max_samples,
        cache_path=real_cache_path,
    )
    print()

    # Step 2: 生成生成数据 embedding
    print("-" * 50)
    print(f"Step 2: 生成 {gen_name} embedding")
    print("-" * 50)
    gen_emb = generate_embeddings(
        data_iterator=gen_iterator,
        field=field,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        max_samples=max_samples,
        cache_path=gen_cache_path,
    )
    print()

    emb_time = time.time() - start_time

    # Step 3: 计算 MMD
    print("-" * 50)
    print("Step 3: 计算 MMD 分布距离")
    print("-" * 50)

    mmd_start = time.time()
    mmd_result = compute_mmd(
        real_features=real_emb,
        gen_features=gen_emb,
        kernel="rbf",
        bandwidths=bandwidths,
    )
    mmd_time = time.time() - mmd_start
    total_time = time.time() - start_time

    results = {
        "real_dataset": real_name,
        "gen_dataset": gen_name,
        "field": field,
        "embedding_model": embedding_model,
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "embedding_time_seconds": emb_time,
        "mmd_time_seconds": mmd_time,
        **mmd_result,
    }

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_file}")

    # 打印摘要
    print()
    print("=" * 70)
    print(f"评估完成！总耗时 {total_time:.1f} 秒")
    print("=" * 70)
    print()
    print(f"  {real_name} ({mmd_result['n_real']:,} 样本) vs "
          f"{gen_name} ({mmd_result['n_gen']:,} 样本)")
    print(f"  MMD²:             {mmd_result['mmd2']:.6f}")
    print(f"  Similarity Score: {mmd_result['similarity_score']:.6f}")
    print()

    return results


__all__ = [
    "compute_mmd",
    "compute_distribution_similarity",
]
