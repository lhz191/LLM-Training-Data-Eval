"""
Tabular Data Evaluation Runner

所有指标均为纯数据层面，仅需 syn + real (+ holdout) 数据。

Quality:
  - Validity:  compute_column_validity, compute_violation_rate
  - Fidelity:  compute_marginal_fidelity, compute_pairwise_fidelity,
               compute_detection_score, compute_alpha_precision
  - Diversity:  compute_beta_recall

Trustworthy:
  - Privacy:   compute_dcr, compute_mia
  - Fairness:  compute_coverage_gap, compute_conditional_shift

Usage:
    python execute.py -f configs/test.yaml
"""

import argparse
import os
import json

import pandas as pd

from configs.basic_cfg import get_cfg

from metrics.validity import compute_column_validity, compute_violation_rate
from metrics.fidelity import (
    compute_marginal_fidelity,
    compute_pairwise_fidelity,
    compute_detection_score,
    compute_alpha_precision,
)
from metrics.diversity import compute_beta_recall
from metrics.privacy import compute_dcr, compute_mia
from metrics.fairness import compute_coverage_gap, compute_conditional_shift


def main(args):
    config = get_cfg(args.config_file)

    syn = pd.read_csv(config.synthetic_path)
    real = pd.read_csv(config.real_path)

    holdout = None
    if config.holdout_path and os.path.exists(config.holdout_path):
        holdout = pd.read_csv(config.holdout_path)

    outputs = {}

    # ── Quality: Validity ───────────────────────────────────
    if config.metrics.validity:
        print("[Validity] Column validity ...")
        outputs['column_validity'] = compute_column_validity(
            syn, real, cs_threshold=config.validity.chi2_threshold,
        )
        if config.validity.constraints:
            print("[Validity] Violation rate ...")
            outputs['violation_rate'] = compute_violation_rate(
                syn, list(config.validity.constraints),
            )

    # ── Quality: Fidelity ───────────────────────────────────
    if config.metrics.fidelity:
        print("[Fidelity] Marginal ...")
        outputs['marginal_fidelity'] = compute_marginal_fidelity(syn, real)

        print("[Fidelity] Pairwise ...")
        outputs['pairwise_fidelity'] = compute_pairwise_fidelity(
            syn, real, mixed_n_bins=config.fidelity.mixed_n_bins,
        )

        print("[Fidelity] Detection score ...")
        outputs['detection_score'] = compute_detection_score(syn, real)

        print("[Fidelity] α-Precision / β-Recall / Authenticity ...")
        outputs['alpha_precision'] = compute_alpha_precision(syn, real)

    # ── Quality: Diversity ──────────────────────────────────
    if config.metrics.diversity:
        print("[Diversity] β-Recall ...")
        outputs['beta_recall'] = compute_beta_recall(syn, real)

    # ── Trustworthy: Privacy ────────────────────────────────
    if config.metrics.privacy:
        print("[Privacy] DCR ...")
        outputs['dcr'] = compute_dcr(syn, real, holdout=holdout)

        if holdout is not None:
            print("[Privacy] MIA ...")
            outputs['mia'] = compute_mia(syn, real, holdout)
        else:
            outputs['mia'] = {'note': 'holdout_path not provided, MIA skipped'}

    # ── Trustworthy: Fairness ───────────────────────────────
    if config.metrics.fairness:
        pa = config.fairness.protected_attribute
        lc = config.fairness.label_column
        if pa:
            print(f"[Fairness] CovGap (attr={pa}) ...")
            outputs['coverage_gap'] = compute_coverage_gap(syn, real, pa)

            if lc:
                print(f"[Fairness] CondShift (attr={pa}, label={lc}) ...")
                outputs['conditional_shift'] = compute_conditional_shift(
                    syn, real, pa, lc,
                )
        else:
            outputs['fairness'] = {'note': 'protected_attribute not set'}

    # ── Save ────────────────────────────────────────────────
    output_path = os.path.join(config.output_dir or '.', "res.json")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False, default=str)

    print(f"\nResults saved to {output_path}")

    for key in ['column_validity', 'marginal_fidelity', 'detection_score',
                'alpha_precision', 'beta_recall', 'dcr']:
        if key in outputs:
            for sk, sv in outputs[key].items():
                if isinstance(sv, float):
                    print(f"  {key}.{sk}: {sv:.4f}")
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, default="")
    args = parser.parse_args()
    main(args)
