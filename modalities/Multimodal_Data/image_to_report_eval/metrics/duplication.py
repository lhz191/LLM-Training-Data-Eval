#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duplication 重复性检测 — Image-to-Report 数据集重复/近似重复检测

从三个层面检测数据集中的冗余：

    1. 精确重复 (Exact Duplicates)
       报告文本完全相同的样本对。直接用 hash 分组。

    2. 近似重复 (Near Duplicates)
       报告文本高度相似但不完全相同的样本对（如仅标点/大小写差异、
       XXXX 占位符不同位置等）。使用 embedding 余弦相似度检测。

    3. 图像重复 (Image Duplicates)
       不同样本引用了相同图像路径。同一张图片出现在多个样本中
       意味着可能存在数据泄露或标注冗余。

使用方式:
    from loaders import IUXRayLoader
    from metrics.duplication import compute_duplication

    loader = IUXRayLoader('/path/to/IU-Xray', split='train')

    results = compute_duplication(
        data_iterator=loader.iterate(),
        dataset_name='IU X-Ray (train)',
        embedding_cache_path='embeddings/iu_xray_report.npy',
        near_dup_threshold=0.95,
    )
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'

import sys
import json
import time
import hashlib
import random
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional, Iterator, Dict, Any, List, Tuple, Set

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data_types import ImageToReportSample


# =============================================================================
# Embedding 复用（与 diversity.py 同一套）
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_SCRIPT_DIR, "models")

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


def _generate_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    cache_path: Optional[str] = None,
) -> np.ndarray:
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
# 1. 精确重复检测
# =============================================================================

def _detect_exact_duplicates(
    samples: List[ImageToReportSample],
) -> Dict[str, Any]:
    """通过 hash 对报告文本分组，找出完全相同的报告"""

    hash_groups: Dict[str, List[str]] = defaultdict(list)
    for s in samples:
        h = hashlib.md5(s.report.encode("utf-8")).hexdigest()
        hash_groups[h].append(s.sample_id)

    dup_groups = {h: ids for h, ids in hash_groups.items() if len(ids) > 1}
    n_dup_samples = sum(len(ids) for ids in dup_groups.values())
    n_dup_groups = len(dup_groups)

    top_groups = []
    for h, ids in sorted(dup_groups.items(), key=lambda x: -len(x[1]))[:20]:
        report_text = ""
        for s in samples:
            if s.sample_id == ids[0]:
                report_text = s.report[:200]
                break
        top_groups.append({
            "count": len(ids),
            "sample_ids": ids[:10],
            "report_preview": report_text,
        })

    return {
        "n_total": len(samples),
        "n_unique_reports": len(hash_groups),
        "n_duplicate_groups": n_dup_groups,
        "n_duplicate_samples": n_dup_samples,
        "duplicate_rate": n_dup_samples / len(samples) if samples else 0.0,
        "top_duplicate_groups": top_groups,
    }


# =============================================================================
# 2. 近似重复检测
# =============================================================================

def _detect_near_duplicates(
    samples: List[ImageToReportSample],
    embeddings: np.ndarray,
    threshold: float = 0.95,
    max_pairs_to_report: int = 50,
    scan_sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    基于 embedding 余弦相似度检测近似重复。
    embedding 已经 L2 归一化，点积即余弦相似度。

    当样本量大时，全量 O(N^2) 不现实。采用分块策略：
    随机采样 scan_sample_size 个样本作为查询，对全集做 KNN 检索。
    """
    n = len(samples)

    if scan_sample_size and scan_sample_size < n:
        random.seed(random_seed)
        query_indices = random.sample(range(n), scan_sample_size)
    else:
        query_indices = list(range(n))
        scan_sample_size = n

    from sklearn.neighbors import NearestNeighbors

    k_neighbors = min(20, n - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine", algorithm="auto", n_jobs=1)
    nn.fit(embeddings)

    query_emb = embeddings[query_indices]
    distances, indices = nn.kneighbors(query_emb)

    near_dup_pairs: List[Tuple[str, str, float]] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for qi, q_idx in enumerate(query_indices):
        for ni in range(1, k_neighbors + 1):
            neighbor_idx = indices[qi][ni]
            cosine_dist = distances[qi][ni]
            similarity = 1.0 - cosine_dist

            if similarity < threshold:
                break

            if q_idx == neighbor_idx:
                continue
            if samples[q_idx].report == samples[neighbor_idx].report:
                continue

            pair_key = tuple(sorted([samples[q_idx].sample_id, samples[neighbor_idx].sample_id]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            near_dup_pairs.append((
                samples[q_idx].sample_id,
                samples[neighbor_idx].sample_id,
                float(similarity),
            ))

    near_dup_pairs.sort(key=lambda x: -x[2])

    reported_pairs = []
    for id_a, id_b, sim in near_dup_pairs[:max_pairs_to_report]:
        text_a = text_b = ""
        for s in samples:
            if s.sample_id == id_a:
                text_a = s.report[:150]
            elif s.sample_id == id_b:
                text_b = s.report[:150]
        reported_pairs.append({
            "id_a": id_a,
            "id_b": id_b,
            "similarity": round(sim, 4),
            "report_a_preview": text_a,
            "report_b_preview": text_b,
        })

    return {
        "threshold": threshold,
        "n_scanned_queries": len(query_indices),
        "n_near_duplicate_pairs": len(near_dup_pairs),
        "near_duplicate_rate": len(near_dup_pairs) / max(len(query_indices), 1),
        "top_near_duplicate_pairs": reported_pairs,
    }


# =============================================================================
# 3. 图像重复检测
# =============================================================================

def _detect_image_duplicates(
    samples: List[ImageToReportSample],
) -> Dict[str, Any]:
    """检测不同样本中引用了相同图像路径的情况"""
    image_to_samples: Dict[str, List[str]] = defaultdict(list)

    for s in samples:
        for img_path in s.images:
            image_to_samples[img_path].append(s.sample_id)

    dup_images = {
        img: ids for img, ids in image_to_samples.items() if len(ids) > 1
    }

    n_total_images = len(image_to_samples)
    n_dup_images = len(dup_images)
    n_samples_with_shared_images = len(set(
        sid for ids in dup_images.values() for sid in ids
    ))

    top_shared = []
    for img, ids in sorted(dup_images.items(), key=lambda x: -len(x[1]))[:20]:
        top_shared.append({
            "image_path": img,
            "n_samples": len(ids),
            "sample_ids": ids[:10],
        })

    return {
        "n_total_unique_images": n_total_images,
        "n_shared_images": n_dup_images,
        "image_sharing_rate": n_dup_images / n_total_images if n_total_images > 0 else 0.0,
        "n_samples_with_shared_images": n_samples_with_shared_images,
        "top_shared_images": top_shared,
    }


# =============================================================================
# 主函数
# =============================================================================

def compute_duplication(
    data_iterator: Iterator[ImageToReportSample],
    dataset_name: str = "Unknown",
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_cache_path: Optional[str] = None,
    embedding_batch_size: int = 64,
    near_dup_threshold: float = 0.95,
    near_dup_scan_size: Optional[int] = 2000,
    output_file: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    计算 Image-to-Report 数据集的重复性指标

    Args:
        data_iterator: ImageToReportSample 迭代器
        dataset_name: 数据集名称
        embedding_model: Embedding 模型名称
        embedding_cache_path: embedding 缓存 .npy 路径（与 diversity 共享）
        embedding_batch_size: embedding 生成 batch size
        near_dup_threshold: 近似重复的余弦相似度阈值
        near_dup_scan_size: 近似重复扫描的查询采样数（None=全量，大数据集建议 2000）
        output_file: 结果保存路径
        max_samples: 最大样本数
    """
    print("=" * 70)
    print("Duplication 重复性检测 (Image-to-Report)")
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print(f"近似重复阈值: {near_dup_threshold}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if max_samples:
        print(f"样本限制: {max_samples}")
    print()

    start_time = time.time()

    # 收集样本
    samples: List[ImageToReportSample] = []
    for s in data_iterator:
        if max_samples and len(samples) >= max_samples:
            break
        samples.append(s)
    print(f"共 {len(samples):,} 条样本")
    print()

    # 1. 精确重复
    print("-" * 50)
    print("Step 1: 精确重复检测")
    print("-" * 50)
    exact_dup = _detect_exact_duplicates(samples)
    print(f"  唯一报告数: {exact_dup['n_unique_reports']:,} / {exact_dup['n_total']:,}")
    print(f"  重复组数: {exact_dup['n_duplicate_groups']}")
    print(f"  涉及重复样本: {exact_dup['n_duplicate_samples']:,} ({exact_dup['duplicate_rate']:.2%})")
    if exact_dup["top_duplicate_groups"]:
        print(f"  最大重复组: {exact_dup['top_duplicate_groups'][0]['count']} 条相同")
        print(f"    预览: {exact_dup['top_duplicate_groups'][0]['report_preview'][:100]}...")
    print()

    # 2. 近似重复
    print("-" * 50)
    print("Step 2: 近似重复检测 (embedding 余弦相似度)")
    print("-" * 50)
    reports = [s.report for s in samples]
    embeddings = _generate_embeddings(
        texts=reports,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        cache_path=embedding_cache_path,
    )
    print(f"  Embedding shape: {embeddings.shape}")
    near_dup = _detect_near_duplicates(
        samples, embeddings,
        threshold=near_dup_threshold,
        scan_sample_size=near_dup_scan_size,
    )
    print(f"  扫描查询数: {near_dup['n_scanned_queries']:,}")
    print(f"  近似重复对数: {near_dup['n_near_duplicate_pairs']}")
    if near_dup["top_near_duplicate_pairs"]:
        top = near_dup["top_near_duplicate_pairs"][0]
        print(f"  最相似一对 (sim={top['similarity']:.4f}):")
        print(f"    A: {top['report_a_preview'][:80]}...")
        print(f"    B: {top['report_b_preview'][:80]}...")
    print()

    # 3. 图像重复
    print("-" * 50)
    print("Step 3: 图像路径重复检测")
    print("-" * 50)
    image_dup = _detect_image_duplicates(samples)
    print(f"  唯一图像路径: {image_dup['n_total_unique_images']:,}")
    print(f"  被多样本共享的图像: {image_dup['n_shared_images']} ({image_dup['image_sharing_rate']:.2%})")
    print(f"  涉及共享图像的样本: {image_dup['n_samples_with_shared_images']}")
    if image_dup["top_shared_images"]:
        top_img = image_dup["top_shared_images"][0]
        print(f"  最多共享: {top_img['image_path']} ({top_img['n_samples']} 个样本)")
    print()

    total_time = time.time() - start_time

    results = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": round(total_time, 2),
        "n_samples": len(samples),
        "exact_duplicates": exact_dup,
        "near_duplicates": near_dup,
        "image_duplicates": image_dup,
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
    print("【1. 精确重复】")
    print(f"  唯一报告:     {exact_dup['n_unique_reports']:,} / {exact_dup['n_total']:,}")
    print(f"  重复率:       {exact_dup['duplicate_rate']:.2%}")
    print()
    print("【2. 近似重复】 (阈值 {:.2f})".format(near_dup_threshold))
    print(f"  近似重复对:   {near_dup['n_near_duplicate_pairs']}")
    print()
    print("【3. 图像路径重复】")
    print(f"  共享图像:     {image_dup['n_shared_images']} / {image_dup['n_total_unique_images']:,}")
    print(f"  共享率:       {image_dup['image_sharing_rate']:.2%}")
    print()

    return results
