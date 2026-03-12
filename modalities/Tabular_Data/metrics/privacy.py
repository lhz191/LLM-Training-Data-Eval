"""
Privacy — DCR, MIA  (Survey §4.3)

仅实现可从生成数据本身计算的隐私指标，不依赖生成机制内部参数，
也不依赖下游模型训练。

1. DCR (Distance to Closest Record):
   - DCR_train→gen: 合成记录到最近训练记录的距离 (记忆化风险)
   - DCR_train→test: 测试记录到最近训练记录的距离 (real baseline)
   - DCR_real→syn: 真实记录到最近合成记录的距离 (覆盖度方向)
   参考: Borisov et al. (2023), Fang et al. (2024)

2. MIA (Membership Inference Attack):
   - AUC_MIA: 攻击者区分成员/非成员的 ROC-AUC
   - Adv_MIA: max_τ (TPR(τ) - FPR(τ))
   攻击分数 g(x) = -min_{s∈D̂} d(x,s)，即到合成数据的负距离
   参考: Shokri et al. (2017), Yeom et al. (2018)

不实现:
  - AIA: 需要在 syn 上训练下游分类器，非数据本身的性质
  - DP (ε, δ): 生成机制属性，不可从输出数据计算

依赖: sklearn, pandas, numpy
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, Any, Optional


# ═══════════════════════════════════════════════════════════
#  编码工具
# ═══════════════════════════════════════════════════════════

def _encode_dataframe(
    df: pd.DataFrame,
    encoders: Optional[dict] = None,
) -> tuple:
    """
    将混合类型 DataFrame 编码为 float 数组。
    返回 (array, encoders_dict)，encoders_dict 可复用于其他 DataFrame。
    """
    df = df.copy()
    if encoders is None:
        encoders = {}
        fit = True
    else:
        fit = False

    for col in df.select_dtypes(include=['object', 'category', 'bool']).columns:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is not None:
                mapping = {v: i for i, v in enumerate(le.classes_)}
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
            else:
                df[col] = 0

    df = df.fillna(0)
    return df.values.astype(float), encoders


# ═══════════════════════════════════════════════════════════
#  1. DCR (Distance to Closest Record)
# ═══════════════════════════════════════════════════════════

def compute_dcr(
    syn: pd.DataFrame,
    train: pd.DataFrame,
    holdout: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    计算三个方向的 DCR:
      - dcr_train_to_gen:  DCR(train→gen)  — 合成记录到训练集的最近距离
      - dcr_train_to_test: DCR(train→test) — 测试记录到训练集的最近距离 (baseline)
      - dcr_real_to_syn:   DCR(real→syn)   — 测试记录到合成集的最近距离 (覆盖度)

    若 holdout 为 None，只计算 dcr_train_to_gen。
    所有特征先做 StandardScaler 归一化。
    """
    common = [c for c in train.columns if c in syn.columns]
    if holdout is not None:
        common = [c for c in common if c in holdout.columns]

    train_enc, encoders = _encode_dataframe(train[common])
    syn_enc, _ = _encode_dataframe(syn[common], encoders=encoders)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_enc)
    syn_scaled = scaler.transform(syn_enc)

    def _dcr_stats(dists: np.ndarray, label: str) -> dict:
        return {
            f'{label}_mean': float(np.mean(dists)),
            f'{label}_median': float(np.median(dists)),
            f'{label}_min': float(np.min(dists)),
            f'{label}_5th': float(np.percentile(dists, 5)),
            f'{label}_exact_match_rate': float((dists < 1e-6).mean()),
        }

    nn_train = NearestNeighbors(n_neighbors=1)
    nn_train.fit(train_scaled)
    dists_gen, _ = nn_train.kneighbors(syn_scaled)
    result = _dcr_stats(dists_gen[:, 0], 'dcr_train_to_gen')

    if holdout is not None and len(holdout) > 0:
        hold_enc, _ = _encode_dataframe(holdout[common], encoders=encoders)
        hold_scaled = scaler.transform(hold_enc)

        dists_test, _ = nn_train.kneighbors(hold_scaled)
        result.update(_dcr_stats(dists_test[:, 0], 'dcr_train_to_test'))

        nn_syn = NearestNeighbors(n_neighbors=1)
        nn_syn.fit(syn_scaled)
        dists_real_syn, _ = nn_syn.kneighbors(hold_scaled)
        result.update(_dcr_stats(dists_real_syn[:, 0], 'dcr_real_to_syn'))

    return result


# ═══════════════════════════════════════════════════════════
#  2. MIA (Membership Inference Attack)
# ═══════════════════════════════════════════════════════════

def compute_mia(
    syn: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> Dict[str, Any]:
    """
    基于距离的成员推断攻击。

    攻击分数: g(x) = -min_{s∈D̂} d(x, s)
    成员(train) 由于被模型记忆，g(x) 倾向更大。

    返回:
      mia_auc:       AUC_MIA = Pr[g(x_train) > g(x_holdout)]
      mia_advantage: Adv_MIA = max_τ (TPR(τ) - FPR(τ))
    """
    common = [c for c in train.columns if c in syn.columns and c in holdout.columns]
    if len(common) == 0:
        return {'mia_auc': 0.5, 'mia_advantage': 0.0, 'note': 'no common columns'}

    train_enc, encoders = _encode_dataframe(train[common])
    hold_enc, _ = _encode_dataframe(holdout[common], encoders=encoders)
    syn_enc, _ = _encode_dataframe(syn[common], encoders=encoders)

    scaler = StandardScaler()
    scaler.fit(np.vstack([train_enc, hold_enc]))
    train_scaled = scaler.transform(train_enc)
    hold_scaled = scaler.transform(hold_enc)
    syn_scaled = scaler.transform(syn_enc)

    nn_syn = NearestNeighbors(n_neighbors=1)
    nn_syn.fit(syn_scaled)

    dists_train, _ = nn_syn.kneighbors(train_scaled)
    dists_hold, _ = nn_syn.kneighbors(hold_scaled)

    g_train = -dists_train[:, 0]
    g_hold = -dists_hold[:, 0]

    scores = np.concatenate([g_train, g_hold])
    labels = np.concatenate([np.ones(len(g_train)), np.zeros(len(g_hold))])

    try:
        auc = float(roc_auc_score(labels, scores))
    except ValueError:
        auc = 0.5

    fpr, tpr, _ = roc_curve(labels, scores)
    advantage = float(np.max(tpr - fpr))

    return {
        'mia_auc': auc,
        'mia_advantage': advantage,
    }
