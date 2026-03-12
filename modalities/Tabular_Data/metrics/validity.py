"""
Validity — Chi-squared Test & Violation Rate  (Survey §4.2)

使用 SDMetrics (sdv-dev/SDMetrics) 进行 marginal 分布检验:
  - CSTest:            分类列 Chi-squared 检验
  - TVComplement:      分类列 Total Variation 互补 (推荐替代 CSTest)
  - KSComplement:      数值列 Kolmogorov-Smirnov 互补
  - BoundaryAdherence: 数值列是否在 real 数据范围内
  - CategoryAdherence: 分类列是否只含 real 中出现过的类别

Violation Rate (VR = #violations / #checks) 使用自定义约束检查。

依赖: sdmetrics, scipy, pandas, numpy
      pip install sdmetrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sdmetrics.single_column import CSTest, TVComplement, KSComplement
from sdmetrics.single_column import BoundaryAdherence, CategoryAdherence


# ═══════════════════════════════════════════════════════════
#  1. Marginal Distribution Tests (via SDMetrics)
# ═══════════════════════════════════════════════════════════

def compute_column_validity(
    syn: pd.DataFrame,
    real: pd.DataFrame,
    cs_threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    逐列检验 synthetic 与 real 分布是否一致。

    数值列: KSComplement (1 = 完美匹配, 0 = 完全不同)
    分类列: CSTest (p-value), TVComplement (1 = 完美匹配)

    同时报告 BoundaryAdherence / CategoryAdherence 作为诊断。
    """
    num_cols = [c for c in real.select_dtypes(include=[np.number]).columns if c in syn.columns]
    cat_cols = [c for c in real.select_dtypes(include=['object', 'category', 'bool']).columns if c in syn.columns]

    results = {}

    # ── 数值列 ──
    ks_scores = {}
    boundary_scores = {}
    for col in num_cols:
        r, s = real[col].dropna(), syn[col].dropna()
        if len(r) == 0 or len(s) == 0:
            continue
        ks_scores[col] = float(KSComplement.compute(real_data=r, synthetic_data=s))
        boundary_scores[col] = float(BoundaryAdherence.compute(real_data=r, synthetic_data=s))

    # ── 分类列 ──
    cs_scores = {}
    tv_scores = {}
    cat_adh_scores = {}
    cs_passed = 0
    for col in cat_cols:
        r, s = real[col].dropna(), syn[col].dropna()
        if len(r) == 0 or len(s) == 0:
            continue
        cs_val = float(CSTest.compute(real_data=r, synthetic_data=s))
        tv_val = float(TVComplement.compute(real_data=r, synthetic_data=s))
        cat_val = float(CategoryAdherence.compute(real_data=r, synthetic_data=s))
        cs_scores[col] = cs_val
        tv_scores[col] = tv_val
        cat_adh_scores[col] = cat_val
        if cs_val >= cs_threshold:
            cs_passed += 1

    results['ks_complement'] = ks_scores
    results['avg_ks_complement'] = float(np.mean(list(ks_scores.values()))) if ks_scores else 0.0
    results['boundary_adherence'] = boundary_scores
    results['avg_boundary_adherence'] = float(np.mean(list(boundary_scores.values()))) if boundary_scores else 0.0

    results['cs_test'] = cs_scores
    results['cs_pass_rate'] = cs_passed / len(cs_scores) if cs_scores else 0.0
    results['tv_complement'] = tv_scores
    results['avg_tv_complement'] = float(np.mean(list(tv_scores.values()))) if tv_scores else 0.0
    results['category_adherence'] = cat_adh_scores
    results['avg_category_adherence'] = float(np.mean(list(cat_adh_scores.values()))) if cat_adh_scores else 0.0

    all_scores = list(ks_scores.values()) + list(tv_scores.values())
    results['overall_validity_score'] = float(np.mean(all_scores)) if all_scores else 0.0

    return results


# ═══════════════════════════════════════════════════════════
#  2. Violation Rate (自定义约束检查)
# ═══════════════════════════════════════════════════════════

def compute_violation_rate(
    syn: pd.DataFrame,
    constraints: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    VR = #violations / #checks

    Each constraint dict:
      - "type": "range" | "unique" | "not_null" | "regex" | "functional_dep"
      - "column": str (or "columns" for functional_dep)
      - type-specific params
    """
    total_checks = 0
    violations = 0
    details = []

    for constraint in constraints:
        ctype = constraint.get('type', '')
        col = constraint.get('column', '')

        if ctype == 'range' and col in syn.columns:
            cmin = constraint.get('min', -np.inf)
            cmax = constraint.get('max', np.inf)
            series = pd.to_numeric(syn[col], errors='coerce')
            n = int(series.notna().sum())
            v = int(((series < cmin) | (series > cmax)).sum())
            total_checks += n
            violations += v
            details.append({'type': ctype, 'column': col, 'violations': v, 'checks': n})

        elif ctype == 'unique' and col in syn.columns:
            n = len(syn)
            v = n - syn[col].nunique()
            total_checks += n
            violations += v
            details.append({'type': ctype, 'column': col, 'violations': v, 'checks': n})

        elif ctype == 'not_null' and col in syn.columns:
            n = len(syn)
            v = int(syn[col].isna().sum())
            total_checks += n
            violations += v
            details.append({'type': ctype, 'column': col, 'violations': v, 'checks': n})

        elif ctype == 'regex' and col in syn.columns:
            import re
            pattern = constraint.get('pattern', '.*')
            series = syn[col].astype(str)
            n = len(series)
            v = sum(1 for s in series if not re.match(pattern, s))
            total_checks += n
            violations += v
            details.append({'type': ctype, 'column': col, 'violations': v, 'checks': n})

        elif ctype == 'functional_dep':
            cols = constraint.get('columns', [])
            target = constraint.get('target', '')
            if len(cols) >= 1 and target and all(c in syn.columns for c in cols + [target]):
                grouped = syn.groupby(cols)[target].nunique()
                v = int((grouped > 1).sum())
                n = int(len(grouped))
                total_checks += n
                violations += v
                details.append({'type': ctype, 'columns': cols, 'target': target, 'violations': v, 'checks': n})

    vr = violations / total_checks if total_checks > 0 else 0.0
    return {
        'violation_rate': vr,
        'total_checks': total_checks,
        'total_violations': violations,
        'details': details,
    }
