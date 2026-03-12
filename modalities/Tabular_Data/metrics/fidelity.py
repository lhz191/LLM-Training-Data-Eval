"""
Fidelity — Marginal, Pairwise, and Global metrics  (Survey §4.2)

使用 SDMetrics (sdv-dev/SDMetrics) 进行统计相似度评估:

Marginal:
  KSComplement:          数值列分布相似度 (1-KS statistic)
  TVComplement:          分类列分布相似度 (1-TVD)

Pairwise:
  CorrelationSimilarity: 数值列对相关系数差异
  ContingencySimilarity: 分类列对联合分布差异

Global:
  LogisticDetection:     分类器二样本检验 (DetScore)

Sample-level:
  α-Precision / β-Recall: synthcity AlphaPrecision (Alaa et al., 2022)
  与 TabSyn (Zhang et al., 2024a) 评测方式完全一致

依赖: sdmetrics, synthcity, sklearn, pandas, numpy
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Dict, Any

from sdmetrics.single_column import KSComplement, TVComplement
from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity
from sdmetrics.single_table import LogisticDetection

from synthcity.metrics.eval_statistical import AlphaPrecision as _SynthcityAlphaPrecision
from synthcity.plugins.core.dataloader import GenericDataLoader


# ═══════════════════════════════════════════════════════════
#  1. Marginal Fidelity (via SDMetrics)
# ═══════════════════════════════════════════════════════════

def compute_marginal_fidelity(
    syn: pd.DataFrame,
    real: pd.DataFrame,
) -> Dict[str, Any]:
    """
    KSComplement (数值列) + TVComplement (分类列)，逐列计算。
    分数 1.0 = 完美匹配，0.0 = 完全不同。
    """
    num_cols = [c for c in real.select_dtypes(include=[np.number]).columns if c in syn.columns]
    cat_cols = [c for c in real.select_dtypes(include=['object', 'category', 'bool']).columns if c in syn.columns]

    ks_results = {}
    for col in num_cols:
        r, s = real[col].dropna(), syn[col].dropna()
        if len(r) == 0 or len(s) == 0:
            continue
        ks_results[col] = float(KSComplement.compute(real_data=r, synthetic_data=s))

    tv_results = {}
    for col in cat_cols:
        r, s = real[col].dropna(), syn[col].dropna()
        if len(r) == 0 or len(s) == 0:
            continue
        tv_results[col] = float(TVComplement.compute(real_data=r, synthetic_data=s))

    avg_ks = float(np.mean(list(ks_results.values()))) if ks_results else 0.0
    avg_tv = float(np.mean(list(tv_results.values()))) if tv_results else 0.0

    return {
        'ks_complement': ks_results,
        'tv_complement': tv_results,
        'avg_ks': avg_ks,
        'avg_tv': avg_tv,
    }


# ═══════════════════════════════════════════════════════════
#  2. Pairwise Fidelity (via SDMetrics)
# ═══════════════════════════════════════════════════════════

def _bin_numerical_column(
    real_series: pd.Series,
    syn_series: pd.Series,
    n_bins: int = 10,
) -> tuple:
    """
    将数值列等频分桶为分类值，用于 mixed-type pair 的 ContingencyScore。
    bin 边界在 real 数据上计算，syn 复用同一套边界。
    """
    try:
        _, bin_edges = pd.qcut(real_series, q=n_bins, retbins=True, duplicates='drop')
    except ValueError:
        _, bin_edges = pd.cut(real_series, bins=n_bins, retbins=True)

    real_binned = pd.cut(real_series, bins=bin_edges, include_lowest=True).astype(str)
    syn_binned = pd.cut(syn_series, bins=bin_edges, include_lowest=True).astype(str)
    return real_binned, syn_binned


def compute_pairwise_fidelity(
    syn: pd.DataFrame,
    real: pd.DataFrame,
    mixed_n_bins: int = 10,
) -> Dict[str, Any]:
    """
    CorrelationSimilarity (数值-数值对) +
    ContingencySimilarity (分类-分类对) +
    ContingencySimilarity (数值-分类 mixed-type 对，数值列先分桶)。

    mixed-type 处理遵循 TabSyn Appendix E.3:
    "For mixed type pairs, TabSyn buckets numerical values into
     categorical bins before computing the corresponding contingency score."

    分数 1.0 = 相关结构完全保留，0.0 = 完全不同。
    """
    num_cols = [c for c in real.select_dtypes(include=[np.number]).columns if c in syn.columns]
    cat_cols = [c for c in real.select_dtypes(include=['object', 'category', 'bool']).columns if c in syn.columns]

    from itertools import combinations, product

    # ── 数值-数值: CorrelationSimilarity ──
    corr_scores = {}
    for c1, c2 in combinations(num_cols, 2):
        r1, r2 = real[[c1, c2]].dropna(), syn[[c1, c2]].dropna()
        if len(r1) < 3 or len(r2) < 3:
            continue
        score = float(CorrelationSimilarity.compute(
            real_data=real[[c1, c2]], synthetic_data=syn[[c1, c2]]))
        corr_scores[f"{c1}__{c2}"] = score

    # ── 分类-分类: ContingencySimilarity ──
    cont_scores = {}
    for c1, c2 in combinations(cat_cols, 2):
        r1, r2 = real[[c1, c2]].dropna(), syn[[c1, c2]].dropna()
        if len(r1) < 3 or len(r2) < 3:
            continue
        score = float(ContingencySimilarity.compute(
            real_data=real[[c1, c2]], synthetic_data=syn[[c1, c2]]))
        cont_scores[f"{c1}__{c2}"] = score

    # ── 数值-分类 mixed-type: 数值列分桶后计算 ContingencySimilarity ──
    mixed_scores = {}
    for num_c, cat_c in product(num_cols, cat_cols):
        pair_real = real[[num_c, cat_c]].dropna()
        pair_syn = syn[[num_c, cat_c]].dropna()
        if len(pair_real) < 3 or len(pair_syn) < 3:
            continue

        real_binned, syn_binned = _bin_numerical_column(
            pair_real[num_c], pair_syn[num_c], n_bins=mixed_n_bins)

        real_mixed = pd.DataFrame({num_c: real_binned, cat_c: pair_real[cat_c].values})
        syn_mixed = pd.DataFrame({num_c: syn_binned, cat_c: pair_syn[cat_c].values})

        score = float(ContingencySimilarity.compute(
            real_data=real_mixed, synthetic_data=syn_mixed))
        mixed_scores[f"{num_c}__{cat_c}"] = score

    avg_corr = float(np.mean(list(corr_scores.values()))) if corr_scores else 0.0
    avg_cont = float(np.mean(list(cont_scores.values()))) if cont_scores else 0.0
    avg_mixed = float(np.mean(list(mixed_scores.values()))) if mixed_scores else 0.0

    return {
        'correlation_similarity': corr_scores,
        'contingency_similarity': cont_scores,
        'mixed_contingency_similarity': mixed_scores,
        'avg_correlation_similarity': avg_corr,
        'avg_contingency_similarity': avg_cont,
        'avg_mixed_contingency_similarity': avg_mixed,
        'num_numerical_pairs': len(corr_scores),
        'num_categorical_pairs': len(cont_scores),
        'num_mixed_pairs': len(mixed_scores),
    }


# ═══════════════════════════════════════════════════════════
#  3. Detection Score (via SDMetrics)
# ═══════════════════════════════════════════════════════════

def compute_detection_score(
    syn: pd.DataFrame,
    real: pd.DataFrame,
    metadata: dict = None,
) -> Dict[str, float]:
    """
    LogisticDetection: 分类器二样本检验。
    分数 1.0 = 无法区分 (好), 0.0 = 轻松区分 (差)。

    metadata 格式见 SDMetrics 文档，若为 None 则自动推断。
    """
    if metadata is None:
        metadata = _infer_metadata(real)

    score = float(LogisticDetection.compute(
        real_data=real, synthetic_data=syn, metadata=metadata))

    return {
        'detection_score': score,
    }


def _infer_metadata(df: pd.DataFrame) -> dict:
    """从 DataFrame 自动推断 SDMetrics 所需的 metadata。"""
    columns = {}
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            columns[col] = {'sdtype': 'numerical'}
        elif df[col].dtype == bool:
            columns[col] = {'sdtype': 'boolean'}
        else:
            columns[col] = {'sdtype': 'categorical'}
    return {'columns': columns}


# ═══════════════════════════════════════════════════════════
#  4. α-Precision / β-Recall  (synthcity, Alaa et al. 2022)
#     与 TabSyn (Zhang et al., 2024a) 评测完全一致
# ═══════════════════════════════════════════════════════════

def _prepare_encoded_loader(
    df: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
    encoder: OneHotEncoder | None = None,
) -> tuple:
    """
    将混合类型 DataFrame 转换为全数值 DataFrame 并包装为
    synthcity GenericDataLoader，与 TabSyn eval_quality.py 一致。

    返回 (GenericDataLoader, fitted_encoder)。
    """
    num_arr = df[num_cols].to_numpy().astype(float) if num_cols else np.empty((len(df), 0))
    cat_arr = df[cat_cols].to_numpy().astype(str) if cat_cols else np.empty((len(df), 0))

    if cat_arr.shape[1] > 0:
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(cat_arr)
        cat_oh = encoder.transform(cat_arr)
    else:
        cat_oh = np.empty((len(df), 0))
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False)

    combined = np.concatenate([num_arr, cat_oh], axis=1)
    loader = GenericDataLoader(pd.DataFrame(combined).astype(float))
    return loader, encoder


def compute_alpha_precision(
    syn: pd.DataFrame,
    real: pd.DataFrame,
) -> Dict[str, float]:
    """
    使用 synthcity.metrics.eval_statistical.AlphaPrecision 计算
    α-Precision（样本质量）和 β-Recall（样本覆盖度），
    完全复现 TabSyn (Zhang et al., 2024a) 的评测流程。

    返回:
      alpha_precision: δ_precision_alpha (naive)
      beta_recall:     δ_coverage_beta  (naive)
      authenticity:    authenticity      (naive)
    """
    common = [c for c in real.columns if c in syn.columns]
    if not common:
        return {'alpha_precision': 0.0, 'beta_recall': 0.0, 'authenticity': 0.0,
                'note': 'no common columns'}

    real_sub = real[common].dropna().reset_index(drop=True)
    syn_sub = syn[common].dropna().reset_index(drop=True)

    if len(real_sub) < 2 or len(syn_sub) < 2:
        return {'alpha_precision': 0.0, 'beta_recall': 0.0, 'authenticity': 0.0,
                'note': 'insufficient data'}

    num_cols = [c for c in real_sub.select_dtypes(include=[np.number]).columns]
    cat_cols = [c for c in real_sub.select_dtypes(include=['object', 'category', 'bool']).columns]

    real_loader, enc = _prepare_encoded_loader(real_sub, num_cols, cat_cols)
    syn_loader, _ = _prepare_encoded_loader(syn_sub, num_cols, cat_cols, encoder=enc)

    evaluator = _SynthcityAlphaPrecision()
    qual_res = evaluator.evaluate(real_loader, syn_loader)

    return {
        'alpha_precision': float(qual_res.get('delta_precision_alpha_naive', 0.0)),
        'beta_recall':     float(qual_res.get('delta_coverage_beta_naive', 0.0)),
        'authenticity':    float(qual_res.get('authenticity_naive', 0.0)),
    }
