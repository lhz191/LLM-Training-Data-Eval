"""
Fairness — 数据层面诊断  (Survey §4.3)

仅实现纯从 syn + real 数据计算的公平性诊断指标:
  - CovGap:   子群体覆盖差距 (群组比例偏移)
  - CondShift: 标签条件偏移 (群组内标签分布 TV 距离)

不实现:
  - ΔSPD, DIR, ΔEO, ΔEOp: 需要在 syn 上训练下游分类器，非数据本身的性质

依赖: pandas, numpy
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


# ═══════════════════════════════════════════════════════════
#  A. 数据层面诊断
# ═══════════════════════════════════════════════════════════

def compute_coverage_gap(
    syn: pd.DataFrame,
    real: pd.DataFrame,
    protected_attr: str,
) -> Dict[str, Any]:
    """
    CovGap = ½ Σ_{g∈A} |p_syn(A=g) - p_real(A=g)|

    衡量合成数据中保护属性的群组比例与真实数据的偏移。
    值域 [0, 1]，0 = 完美匹配。
    """
    if protected_attr not in syn.columns or protected_attr not in real.columns:
        return {'coverage_gap': 0.0, 'note': f'{protected_attr!r} not found'}

    syn_dist = syn[protected_attr].value_counts(normalize=True)
    real_dist = real[protected_attr].value_counts(normalize=True)

    all_groups = set(syn_dist.index) | set(real_dist.index)
    gap = 0.5 * sum(
        abs(syn_dist.get(g, 0.0) - real_dist.get(g, 0.0))
        for g in all_groups
    )

    per_group = {
        str(g): {
            'p_syn': float(syn_dist.get(g, 0.0)),
            'p_real': float(real_dist.get(g, 0.0)),
        }
        for g in sorted(all_groups, key=str)
    }

    return {
        'coverage_gap': float(gap),
        'per_group': per_group,
    }


def compute_conditional_shift(
    syn: pd.DataFrame,
    real: pd.DataFrame,
    protected_attr: str,
    label_col: str,
) -> Dict[str, Any]:
    """
    CondShift = (1/|A|) Σ_{g∈A} TV(p_syn(Y|A=g), p_real(Y|A=g))

    衡量各群组内标签分布在合成数据与真实数据之间的偏移。
    TV = ½ Σ_y |p(y|g)_syn - p(y|g)_real|，值域 [0, 1]。
    """
    for col in [protected_attr, label_col]:
        if col not in syn.columns or col not in real.columns:
            return {'conditional_shift': 0.0, 'note': f'{col!r} not found'}

    groups = set(syn[protected_attr].unique()) | set(real[protected_attr].unique())
    tvs = {}

    for g in groups:
        syn_g = syn[syn[protected_attr] == g][label_col]
        real_g = real[real[protected_attr] == g][label_col]

        if len(syn_g) == 0 or len(real_g) == 0:
            tvs[str(g)] = 1.0
            continue

        syn_p = syn_g.value_counts(normalize=True)
        real_p = real_g.value_counts(normalize=True)
        all_labels = set(syn_p.index) | set(real_p.index)

        tv = 0.5 * sum(
            abs(syn_p.get(y, 0.0) - real_p.get(y, 0.0))
            for y in all_labels
        )
        tvs[str(g)] = float(tv)

    cond_shift = float(np.mean(list(tvs.values()))) if tvs else 0.0

    return {
        'conditional_shift': cond_shift,
        'per_group_tv': tvs,
    }
