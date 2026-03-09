#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template Divergence 模板偏离度 / 泛化程度

衡量生成数据偏离模板的程度：
- Divergence = 1 - cos(φ(gen), φ(template))
- 适中最好：太低 = 过度复制模板；太高 = 偏离任务语义

使用方式:
    from template_divergence import compute_template_divergence

    results = compute_template_divergence(
        gen_iterator=gen_loader.iterate(),
        template_iterator=template_loader.iterate(),
        gen_name='MyGenData',
        template_name='SeedTemplates',
    )

gen_iterator 和 template_iterator 需要一一对应。
模板只有一条时自动广播到所有生成样本。
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, Iterator, List

import numpy as np
from tqdm import tqdm

import sys
_metrics_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_api_agent_dir = os.path.dirname(_metrics_dir)
sys.path.insert(0, _api_agent_dir)
sys.path.insert(0, _metrics_dir)

from data_types import APIAgentSample  # noqa: E402


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def _collect_text_pairs(
    gen_iterator: Iterator[APIAgentSample],
    template_iterator: Iterator[APIAgentSample],
    max_samples: Optional[int] = None,
) -> tuple:
    """收集生成文本和模板文本的配对列表"""
    gen_texts: List[str] = []
    template_texts: List[str] = []

    template_list = list(template_iterator)
    is_broadcast = len(template_list) == 1

    n = 0
    for sample in tqdm(gen_iterator, desc="收集生成文本"):
        if max_samples is not None and n >= max_samples:
            break
        gen_texts.append((sample.query or "").strip())
        n += 1

    if is_broadcast:
        single_template = (template_list[0].query or "").strip()
        template_texts = [single_template] * len(gen_texts)
        print(f"  模板只有 1 条，广播到 {len(gen_texts)} 条生成数据")
    else:
        for i, sample in enumerate(template_list):
            if i >= len(gen_texts):
                break
            template_texts.append((sample.query or "").strip())
        if len(template_texts) < len(gen_texts):
            gen_texts = gen_texts[:len(template_texts)]

    return gen_texts, template_texts


def compute_template_divergence(
    gen_iterator: Iterator[APIAgentSample],
    template_iterator: Iterator[APIAgentSample],
    gen_name: str = "Generated",
    template_name: str = "Template",
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_batch_size: int = 64,
    max_samples: Optional[int] = None,
    gen_cache_path: Optional[str] = None,
    template_cache_path: Optional[str] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    计算生成数据相对于模板的偏离度（泛化程度）。

    Divergence = 1 - EmbeddingSimilarity。
    适中最好：太低说明过度复制模板，太高说明偏离任务语义。

    Returns:
        {
            "divergence": stats,
            "high_divergence_gt_0.5": int,
            "low_divergence_lt_0.1": int,
        }
    """
    from diversity import generate_embeddings

    print("=" * 70)
    print("Template Divergence 模板偏离度评估")
    print("=" * 70)
    print(f"生成数据: {gen_name} | 模板数据: {template_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    gen_texts, template_texts = _collect_text_pairs(gen_iterator, template_iterator, max_samples)
    n_pairs = len(gen_texts)
    print(f"  有效配对数: {n_pairs}")
    print()

    if n_pairs == 0:
        return {"gen_dataset": gen_name, "template_dataset": template_name, "n_pairs": 0}

    # Embedding
    print("-" * 50)
    print("生成 Embedding 并计算偏离度")
    print("-" * 50)
    gen_samples = [APIAgentSample(query=t, tools=[], api_calls=[]) for t in gen_texts]
    tmpl_samples = [APIAgentSample(query=t, tools=[], api_calls=[]) for t in template_texts]

    gen_emb = generate_embeddings(
        data_iterator=iter(gen_samples), field="query",
        model_name=embedding_model, batch_size=embedding_batch_size,
        cache_path=gen_cache_path,
    )
    tmpl_emb = generate_embeddings(
        data_iterator=iter(tmpl_samples), field="query",
        model_name=embedding_model, batch_size=embedding_batch_size,
        cache_path=template_cache_path,
    )

    gen_norm = gen_emb / np.maximum(np.linalg.norm(gen_emb, axis=1, keepdims=True), 1e-8)
    tmpl_norm = tmpl_emb / np.maximum(np.linalg.norm(tmpl_emb, axis=1, keepdims=True), 1e-8)
    sims = np.sum(gen_norm * tmpl_norm, axis=1)
    div_values = (1.0 - sims).tolist()

    total_time = time.time() - start_time

    n_high = int((sims < 0.5).sum())
    n_low = int((sims >= 0.9).sum())

    results = {
        "gen_dataset": gen_name,
        "template_dataset": template_name,
        "embedding_model": embedding_model,
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "n_pairs": n_pairs,
        "divergence": _summary_stats(div_values),
        "high_divergence_gt_0.5": n_high,
        "low_divergence_lt_0.1": n_low,
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
    print(f"评估完成！耗时 {total_time:.1f} 秒 | 配对数: {n_pairs}")
    print("=" * 70)
    print(f"  平均偏离度: {np.mean(div_values):.4f}  (适中最好)")
    print(f"    高偏离 (>0.5): {n_high} ({100*n_high/n_pairs:.1f}%)")
    print(f"    低偏离 (<0.1): {n_low} ({100*n_low/n_pairs:.1f}%)")
    print()

    return results
