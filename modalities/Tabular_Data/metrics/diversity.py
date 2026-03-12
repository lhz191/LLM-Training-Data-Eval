"""
Diversity — β-Recall  (Survey §4.2, Alaa et al. 2022)

β-Recall measures how well synthetic data covers the real distribution:

    S_syn^β = argmin_S |S|  s.t. Pr(X_syn ∈ S) = β
    R_β = Pr_{X_real ~ P_real}(X_real ∈ S_syn^β)

使用 synthcity.metrics.eval_statistical.AlphaPrecision (naive)，
与 TabSyn (Zhang et al., 2024a) 评测方式完全一致。

synthcity 的 AlphaPrecision 同时计算 α-Precision、β-Recall 和 Authenticity，
此模块复用 fidelity.compute_alpha_precision 以避免重复计算。

依赖: synthcity, sklearn, pandas, numpy (通过 fidelity 模块间接依赖)
"""

import pandas as pd
from typing import Dict, Any

from .fidelity import compute_alpha_precision


def compute_beta_recall(
    syn: pd.DataFrame,
    real: pd.DataFrame,
) -> Dict[str, float]:
    """
    β-Recall (δ_coverage_beta_naive): real 样本落在 synthetic 数据
    β-support 内的比例，衡量合成数据对真实分布的覆盖度。

    内部调用 fidelity.compute_alpha_precision，该函数使用 synthcity
    AlphaPrecision 同时返回 alpha_precision / beta_recall / authenticity。
    """
    result = compute_alpha_precision(syn=syn, real=real)

    return {
        'beta_recall': result['beta_recall'],
    }
