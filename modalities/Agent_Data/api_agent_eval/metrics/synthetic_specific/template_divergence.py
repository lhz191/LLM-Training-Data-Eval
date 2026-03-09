#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template Divergence 模板偏离度 / 泛化程度

衡量生成数据偏离模板的程度，对 query, tools, api_calls 三个字段分别计算：

    Divergence = 1 - cos(φ(gen), φ(template))

适中最好：太低 = 过度复制模板；太高 = 偏离任务语义。

使用方式:
    from template_divergence import compute_template_divergence

    results = compute_template_divergence(
        gen_iterator=gen_loader.iterate(),
        template_iterator=template_loader.iterate(),
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

FIELDS = ["query", "tools", "api_calls"]


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


def _cosine_sim_paired(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐行余弦相似度"""
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return np.sum(a_norm * b_norm, axis=1)


def compute_template_divergence(
    gen_iterator: Iterator[APIAgentSample],
    template_iterator: Iterator[APIAgentSample],
    gen_name: str = "Generated",
    template_name: str = "Template",
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_batch_size: int = 64,
    max_samples: Optional[int] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    计算生成数据相对于模板的偏离度（query/tools/api_calls 三字段）。

    Divergence = 1 - EmbeddingSimilarity。
    适中最好：太低说明过度复制模板，太高说明偏离任务语义。

    Returns:
        {
            "per_field": {
                "query": {"divergence": stats, "high_divergence_gt_0.5": int, ...},
                "tools": {...},
                "api_calls": {...},
            },
            ...
        }
    """
    from diversity import generate_embeddings

    print("=" * 70)
    print("Template Divergence 模板偏离度评估")
    print("=" * 70)
    print(f"生成数据: {gen_name} | 模板数据: {template_name}")
    print(f"字段: {', '.join(FIELDS)}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    # 收集样本
    print("-" * 50)
    print("收集样本")
    print("-" * 50)
    template_list = list(template_iterator)
    is_broadcast = len(template_list) == 1

    gen_list: List[APIAgentSample] = []
    for sample in tqdm(gen_iterator, desc="收集生成样本"):
        if max_samples is not None and len(gen_list) >= max_samples:
            break
        gen_list.append(sample)

    if is_broadcast:
        tmpl_list = template_list * len(gen_list)
        print(f"  模板只有 1 条，广播到 {len(gen_list)} 条生成数据")
    else:
        tmpl_list = template_list[:len(gen_list)]
        if len(tmpl_list) < len(gen_list):
            gen_list = gen_list[:len(tmpl_list)]

    n_pairs = len(gen_list)
    print(f"  有效配对数: {n_pairs}")
    print()

    if n_pairs == 0:
        return {"gen_dataset": gen_name, "template_dataset": template_name, "n_pairs": 0}

    # 对每个字段计算偏离度
    per_field = {}
    for f in FIELDS:
        print("-" * 50)
        print(f"字段: {f}")
        print("-" * 50)

        gen_emb = generate_embeddings(
            data_iterator=iter(gen_list), field=f,
            model_name=embedding_model, batch_size=embedding_batch_size,
        )
        tmpl_emb = generate_embeddings(
            data_iterator=iter(tmpl_list), field=f,
            model_name=embedding_model, batch_size=embedding_batch_size,
        )

        sims = _cosine_sim_paired(gen_emb, tmpl_emb)
        div_values = (1.0 - sims).tolist()

        field_result = {
            "divergence": _summary_stats(div_values),
            "high_divergence_gt_0.5": int((sims < 0.5).sum()),
            "low_divergence_lt_0.1": int((sims >= 0.9).sum()),
        }
        per_field[f] = field_result
        print(f"  平均偏离度: {np.mean(div_values):.4f}")
        print()

    total_time = time.time() - start_time

    results = {
        "gen_dataset": gen_name,
        "template_dataset": template_name,
        "embedding_model": embedding_model,
        "fields": FIELDS,
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "n_pairs": n_pairs,
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
    print(f"评估完成！耗时 {total_time:.1f} 秒 | 配对数: {n_pairs}")
    print("=" * 70)
    for f in FIELDS:
        div_mean = per_field[f]["divergence"]["mean"]
        n_high = per_field[f]["high_divergence_gt_0.5"]
        print(f"  {f:12s}  偏离度: {div_mean:.4f}  (>0.5: {n_high}/{n_pairs})")
    print()

    return results
