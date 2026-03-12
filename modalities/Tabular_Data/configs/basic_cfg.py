"""
Config for Tabular Data Evaluation

Quality:
  - Validity:  Chi-squared, Violation Rate
  - Fidelity:  KST, TVD, PearsonScore, ContingencyScore, Mixed-type, DetScore, α-Precision
  - Diversity:  β-Recall

Trustworthy:
  - Privacy:   DCR, MIA
  - Fairness:  CovGap, CondShift
"""

import os
from yacs.config import CfgNode as CN

_C = CN()

_C.synthetic_path = ''
_C.real_path = ''
_C.holdout_path = ''
_C.output_dir = ''

# ── 指标开关 ──────────────────────────────────────────────
_C.metrics = CN()
_C.metrics.validity = True
_C.metrics.fidelity = True
_C.metrics.diversity = True
_C.metrics.privacy = True
_C.metrics.fairness = False

# ── Validity ──────────────────────────────────────────────
_C.validity = CN()
_C.validity.chi2_threshold = 0.95
_C.validity.constraints = []

# ── Fidelity ──────────────────────────────────────────────
_C.fidelity = CN()
_C.fidelity.mixed_n_bins = 10

# ── Privacy ───────────────────────────────────────────────
_C.privacy = CN()

# ── Fairness ──────────────────────────────────────────────
_C.fairness = CN()
_C.fairness.protected_attribute = ''
_C.fairness.label_column = ''


def get_cfg(config_file_path):
    config = _C.clone()
    if config_file_path:
        config.merge_from_file(config_file_path)
    if config.output_dir and not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)
    return config
