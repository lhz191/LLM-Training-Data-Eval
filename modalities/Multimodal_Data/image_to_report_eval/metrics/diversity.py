#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diversity 多样性指标 — Image-to-Report 数据集多维度多样性评估

从五个维度评估数据集的多样性：

    1. Embedding 语义多样性
       对 report 文本生成 embedding，用 Vendi Score 或 KNN 平均距离衡量
       语义空间中的分散程度。数值越高说明报告之间语义差异越大。

    2. 表达模式多样性 (Self-BLEU)
       对所有 report 计算 Self-BLEU。Self-BLEU 高 → 报告措辞高度雷同（模板化）；
       Self-BLEU 低 → 表达多样。对检测合成数据的模板化问题尤其有效。

    3. 词汇多样性 (Vocabulary)
       Type-Token Ratio (TTR)、Distinct-N (n=1,2,3) 衡量词汇丰富度。
       TTR 低 / Distinct-N 低 → 数据集用词单一。

    4. 报告长度分布
       统计报告长度（字符数和词数）的分布特征。长度方差过小提示数据集
       被截断或模板化生成。

    5. 来源/元数据分布
       如果 metadata 中含有 source / caption_source 等字段，分析其分布均匀度。

使用方式:
    from loaders import IUXRayLoader
    from metrics.diversity import compute_diversity

    loader = IUXRayLoader('/path/to/IU-Xray', split='train')

    results = compute_diversity(
        data_iterator=loader.iterate(),
        dataset_name='IU X-Ray (train)',
        method='knn',
        embedding_cache_path='embeddings/iu_xray_report.npy',
    )
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'

import sys
import json
import math
import time
import random
import numpy as np
from collections import Counter
from datetime import datetime
from typing import Optional, Iterator, Dict, Any, List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import ImageToReportSample


# =============================================================================
# Embedding 模型配置（与 api_agent_eval 共享同一套）
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "models")

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "type": "sentence-transformers",
        "dim": 384,
        "local_path": os.path.join(_MODELS_DIR, "all-MiniLM-L6-v2"),
    },
    "all-mpnet-base-v2": {
        "type": "sentence-transformers",
        "dim": 768,
        "local_path": os.path.join(_MODELS_DIR, "all-mpnet-base-v2"),
    },
}


def _get_model_path(model_name: str) -> str:
    if model_name in EMBEDDING_MODELS:
        local_path = EMBEDDING_MODELS[model_name].get("local_path")
        if local_path and os.path.exists(local_path):
            return local_path
    if os.path.exists(model_name):
        return model_name
    return model_name


# =============================================================================
# Embedding 生成
# =============================================================================

def _generate_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """生成文本 embedding，支持缓存"""
    if cache_path and os.path.exists(cache_path):
        print(f"  从缓存加载 embedding: {cache_path}")
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer
    import torch

    model_path = _get_model_path(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  加载模型: {model_path} ({device})")
    model = SentenceTransformer(model_path, device=device)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        np.save(cache_path, embeddings)
        print(f"  Embedding 已缓存到: {cache_path}")

    return embeddings


# =============================================================================
# Vendi Score
# =============================================================================

def _compute_vendi_score(
    embeddings: np.ndarray,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Vendi Score = exp(entropy of eigenvalues of similarity matrix)"""
    import torch

    n_total = embeddings.shape[0]

    if sample_size and sample_size < n_total:
        random.seed(random_seed)
        indices = random.sample(range(n_total), sample_size)
        embeddings = embeddings[indices]
        is_sampled = True
    else:
        sample_size = n_total
        is_sampled = False

    n = embeddings.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_t = torch.tensor(embeddings, dtype=torch.float32, device=device)

    K = emb_t @ emb_t.T
    K = K / n

    eigenvalues = torch.linalg.eigvalsh(K)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues)).item()
    vendi_score = float(np.exp(entropy))

    del emb_t, K, eigenvalues
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "vendi_score": vendi_score,
        "entropy": float(entropy),
        "n_samples": n,
        "n_total": n_total,
        "is_sampled": is_sampled,
    }


# =============================================================================
# KNN 平均距离
# =============================================================================

def _compute_knn_diversity(
    embeddings: np.ndarray,
    k: int = 10,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors

    n_total = embeddings.shape[0]

    if sample_size and sample_size < n_total:
        random.seed(random_seed)
        indices = random.sample(range(n_total), sample_size)
        embeddings = embeddings[indices]
        is_sampled = True
    else:
        sample_size = n_total
        is_sampled = False

    n = embeddings.shape[0]
    if k >= n:
        k = n - 1

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="auto", n_jobs=1)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    k_distances = distances[:, 1:k + 1]

    return {
        "knn_mean_distance": float(np.mean(k_distances)),
        "knn_std_distance": float(np.std(k_distances)),
        "knn_median_distance": float(np.median(k_distances)),
        "k": k,
        "n_samples": n,
        "n_total": n_total,
        "is_sampled": is_sampled,
    }


# =============================================================================
# Self-BLEU 表达多样性
# =============================================================================

def _compute_self_bleu(
    texts: List[str],
    max_eval: int = 500,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Self-BLEU: 每条 report 以其余 report 为参考集计算 BLEU-4，取均值。
    Self-BLEU 高→措辞雷同；低→表达多样。

    为控制计算量，当样本量大时从全集中抽 max_eval 条作为 hypothesis，
    references 也从全集抽样（最多取 200 条）。
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    n = len(texts)
    if n < 2:
        return {"self_bleu": 1.0, "expression_diversity": 0.0, "n_texts": n}

    tokenized = [t.lower().split() for t in texts]
    smoothing = SmoothingFunction().method1

    random.seed(random_seed)

    if n > max_eval:
        eval_indices = random.sample(range(n), max_eval)
    else:
        eval_indices = list(range(n))

    ref_pool_size = min(n, 200)

    scores = []
    for i in eval_indices:
        hyp = tokenized[i]
        if not hyp:
            continue
        ref_indices = [j for j in random.sample(range(n), ref_pool_size) if j != i]
        refs = [tokenized[j] for j in ref_indices if tokenized[j]]
        if refs:
            scores.append(sentence_bleu(refs, hyp, smoothing_function=smoothing))

    self_bleu = float(np.mean(scores)) if scores else 1.0

    return {
        "self_bleu": self_bleu,
        "expression_diversity": 1.0 - self_bleu,
        "n_evaluated": len(scores),
        "n_texts": n,
    }


# =============================================================================
# 词汇多样性 (TTR + Distinct-N)
# =============================================================================

def _compute_vocabulary_diversity(texts: List[str]) -> Dict[str, Any]:
    """
    Type-Token Ratio (TTR) 和 Distinct-N。
    TTR = 唯一词数 / 总词数；Distinct-N = 唯一 N-gram 数 / 总 N-gram 数。
    """
    all_tokens: List[str] = []
    for t in texts:
        all_tokens.extend(t.lower().split())

    total = len(all_tokens)
    if total == 0:
        return {"ttr": 0.0, "distinct_1": 0.0, "distinct_2": 0.0, "distinct_3": 0.0,
                "total_tokens": 0, "unique_tokens": 0}

    unique = len(set(all_tokens))
    ttr = unique / total

    def _distinct_n(tokens: List[str], n: int) -> float:
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    return {
        "ttr": float(ttr),
        "distinct_1": float(_distinct_n(all_tokens, 1)),
        "distinct_2": float(_distinct_n(all_tokens, 2)),
        "distinct_3": float(_distinct_n(all_tokens, 3)),
        "total_tokens": total,
        "unique_tokens": unique,
    }


# =============================================================================
# 报告长度分布
# =============================================================================

def _compute_length_distribution(texts: List[str]) -> Dict[str, Any]:
    char_lens = [len(t) for t in texts]
    word_lens = [len(t.split()) for t in texts]

    def _stats(arr):
        a = np.array(arr, dtype=float)
        return {
            "min": float(np.min(a)),
            "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "p75": float(np.percentile(a, 75)),
            "max": float(np.max(a)),
            "std": float(np.std(a)),
            "cv": float(np.std(a) / np.mean(a)) if np.mean(a) > 0 else 0.0,
        }

    return {
        "char_length": _stats(char_lens),
        "word_length": _stats(word_lens),
        "n_texts": len(texts),
    }


# =============================================================================
# 来源/元数据分布
# =============================================================================

def _compute_entropy(counter: Counter) -> Tuple[float, float, int]:
    if not counter:
        return 0.0, 0.0, 0
    n = len(counter)
    if n < 2:
        return 0.0, 0.0, n
    total = sum(counter.values())
    if total == 0:
        return 0.0, 0.0, n
    entropy = -sum((c / total) * math.log(c / total) for c in counter.values() if c > 0)
    return entropy / math.log(n), entropy, n


def _compute_gini(values: List[float]) -> float:
    if not values or len(values) < 2:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2 * weighted_sum) / (n * total) - (n + 1) / n


def _compute_metadata_diversity(samples: List[ImageToReportSample]) -> Dict[str, Any]:
    """分析 metadata 中 source / caption_source 等字段的分布"""
    field_counters: Dict[str, Counter] = {}

    interesting_keys = ["source", "caption_source", "subset", "domain", "category"]
    for sample in samples:
        meta = sample.metadata or {}
        for key in interesting_keys:
            val = meta.get(key)
            if val:
                if key not in field_counters:
                    field_counters[key] = Counter()
                field_counters[key][str(val)] += 1

    results = {}
    for key, counter in field_counters.items():
        ent_norm, _, n_unique = _compute_entropy(counter)
        gini = _compute_gini(list(counter.values()))
        results[key] = {
            "n_unique": n_unique,
            "entropy_normalized": ent_norm,
            "gini": gini,
            "distribution": dict(counter.most_common(30)),
        }

    return results if results else {"_note": "metadata 中无 source/caption_source 等字段"}


# =============================================================================
# 主函数
# =============================================================================

def compute_diversity(
    data_iterator: Iterator[ImageToReportSample],
    dataset_name: str = "Unknown",
    method: str = "knn",
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_cache_path: Optional[str] = None,
    embedding_batch_size: int = 64,
    sample_size: Optional[int] = None,
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
    k: int = 10,
    self_bleu_max_eval: int = 500,
) -> Dict[str, Any]:
    """
    计算 Image-to-Report 数据集的多样性指标

    Args:
        data_iterator: ImageToReportSample 迭代器
        dataset_name: 数据集名称
        method: 语义多样性计算方法 ('knn' 或 'vendi')
        embedding_model: Embedding 模型名称
        embedding_cache_path: embedding 缓存 .npy 路径
        embedding_batch_size: embedding 生成 batch size
        sample_size: Vendi/KNN 采样大小（None=全量）
        output_file: 结果保存路径
        max_samples: 最大样本数（用于测试）
        k: KNN 的 K 值
        self_bleu_max_eval: Self-BLEU 评估上限条数
    """
    print("=" * 70)
    print("Diversity 多样性评估 (Image-to-Report)")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"方法: {method} | 模型: {embedding_model}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    print()

    start_time = time.time()

    # Step 0: 收集样本
    print("-" * 50)
    print("Step 0: 收集样本")
    print("-" * 50)
    samples: List[ImageToReportSample] = []
    for s in data_iterator:
        if max_samples and len(samples) >= max_samples:
            break
        samples.append(s)
    print(f"共 {len(samples):,} 条样本")
    print()

    reports = [s.report for s in samples]

    # Step 1: Embedding 语义多样性
    print("-" * 50)
    print("Step 1: Embedding 语义多样性")
    print("-" * 50)

    embeddings = _generate_embeddings(
        texts=reports,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        cache_path=embedding_cache_path,
    )
    print(f"  Embedding shape: {embeddings.shape}")

    if method == "knn":
        semantic_div = _compute_knn_diversity(embeddings, k=k, sample_size=sample_size)
        semantic_score = semantic_div["knn_mean_distance"]
        print(f"  KNN 平均距离: {semantic_score:.6f}")
    elif method == "vendi":
        semantic_div = _compute_vendi_score(embeddings, sample_size=sample_size)
        semantic_score = semantic_div["vendi_score"]
        print(f"  Vendi Score: {semantic_score:.4f}")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'knn' or 'vendi'.")
    print()

    # Step 2: Self-BLEU 表达多样性
    print("-" * 50)
    print("Step 2: Self-BLEU 表达多样性")
    print("-" * 50)
    expression_div = _compute_self_bleu(reports, max_eval=self_bleu_max_eval)
    print(f"  Self-BLEU: {expression_div['self_bleu']:.4f}")
    print(f"  表达多样性: {expression_div['expression_diversity']:.4f}")
    print()

    # Step 3: 词汇多样性
    print("-" * 50)
    print("Step 3: 词汇多样性")
    print("-" * 50)
    vocab_div = _compute_vocabulary_diversity(reports)
    print(f"  TTR: {vocab_div['ttr']:.4f}")
    print(f"  Distinct-1: {vocab_div['distinct_1']:.4f}")
    print(f"  Distinct-2: {vocab_div['distinct_2']:.4f}")
    print(f"  Distinct-3: {vocab_div['distinct_3']:.4f}")
    print(f"  总词数: {vocab_div['total_tokens']:,}, 唯一词数: {vocab_div['unique_tokens']:,}")
    print()

    # Step 4: 报告长度分布
    print("-" * 50)
    print("Step 4: 报告长度分布")
    print("-" * 50)
    length_div = _compute_length_distribution(reports)
    wl = length_div["word_length"]
    print(f"  词数: min={wl['min']:.0f}, median={wl['median']:.0f}, "
          f"mean={wl['mean']:.1f}, max={wl['max']:.0f}, CV={wl['cv']:.2f}")
    cl = length_div["char_length"]
    print(f"  字符: min={cl['min']:.0f}, median={cl['median']:.0f}, "
          f"mean={cl['mean']:.1f}, max={cl['max']:.0f}, CV={cl['cv']:.2f}")
    print()

    # Step 5: 来源/元数据分布
    print("-" * 50)
    print("Step 5: 来源/元数据分布")
    print("-" * 50)
    meta_div = _compute_metadata_diversity(samples)
    if "_note" in meta_div:
        print(f"  {meta_div['_note']}")
    else:
        for key, info in meta_div.items():
            print(f"  {key}: {info['n_unique']} 种, "
                  f"熵(归一化)={info['entropy_normalized']:.4f}, Gini={info['gini']:.4f}")
    print()

    total_time = time.time() - start_time

    # 汇总
    results = {
        "dataset": dataset_name,
        "method": method,
        "embedding_model": embedding_model,
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": round(total_time, 2),
        "n_samples": len(samples),
        "semantic_diversity_score": semantic_score,
        "semantic_diversity": semantic_div,
        "expression_diversity": expression_div,
        "vocabulary_diversity": vocab_div,
        "length_distribution": length_div,
        "metadata_diversity": meta_div,
    }

    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_file}")

    # 摘要
    print()
    print("=" * 70)
    print(f"评估完成！总耗时 {total_time:.1f} 秒")
    print("=" * 70)
    print()
    print(f"数据集: {dataset_name} | 样本数: {len(samples):,}")
    print()
    print("【1. Embedding 语义多样性】")
    if method == "knn":
        print(f"  KNN 平均距离:    {semantic_score:.6f}")
    else:
        print(f"  Vendi Score:     {semantic_score:.4f}")
    print()
    print("【2. 表达模式多样性 (Self-BLEU)】")
    print(f"  Self-BLEU:       {expression_div['self_bleu']:.4f}")
    print(f"  表达多样性:      {expression_div['expression_diversity']:.4f}")
    print()
    print("【3. 词汇多样性】")
    print(f"  TTR:             {vocab_div['ttr']:.4f}")
    print(f"  Distinct-1/2/3:  {vocab_div['distinct_1']:.4f} / "
          f"{vocab_div['distinct_2']:.4f} / {vocab_div['distinct_3']:.4f}")
    print()
    print("【4. 报告长度】")
    print(f"  词数:            median={wl['median']:.0f}, mean={wl['mean']:.1f}, CV={wl['cv']:.2f}")
    print()
    print("【5. 来源分布】")
    if "_note" in meta_div:
        print(f"  {meta_div['_note']}")
    else:
        for key, info in meta_div.items():
            print(f"  {key}: {info['n_unique']} 种, 熵={info['entropy_normalized']:.4f}")
    print()

    return results
