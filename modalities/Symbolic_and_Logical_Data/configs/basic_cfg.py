"""
Config for Symbolic & Logical Data Evaluation

Data loading:
  dataset   — openmath / lila / metamath / gsm8k-aug / numinamath
  data_path — path to dataset file

Quality:
  Validity    — Acc_verify, PassRate, Acc_proof, Consistency
  Fidelity    — Acc_SC, Agree, ρ_LLM-human, Q_reason

Trustworthy:
  Faithfulness — Val_step, Align_entail
  Robustness   — OOD Gap
"""

import os
from yacs.config import CfgNode as CN

_C = CN()

# ══════════════════════════════════════════════════════════
# Data loading — uses math_eval loaders + executors
# ══════════════════════════════════════════════════════════
_C.dataset = ""            # openmath / lila / metamath / gsm8k-aug / numinamath
_C.data_path = ""          # path to dataset file
_C.executor_type = ""      # executor to use (defaults to same as dataset)

_C.jsonl_path = ""         # legacy: raw JSONL (bypasses loader/executor)
_C.output_dir = "eval_res"

# ══════════════════════════════════════════════════════════
# Metric-family toggles
# ══════════════════════════════════════════════════════════
_C.metrics = CN()

_C.metrics.validity = True
_C.metrics.fidelity = True
_C.metrics.faithfulness = True
_C.metrics.robustness = False

# ── Validity sub-toggles ────────────────────────────────
_C.validity = CN()
_C.validity.acc_verify = True
_C.validity.pass_rate = True
_C.validity.acc_proof = True
_C.validity.consistency = True

# ── Fidelity sub-toggles ────────────────────────────────
_C.fidelity = CN()
_C.fidelity.acc_sc = True
_C.fidelity.agreement = True
_C.fidelity.llm_human_correlation = True
_C.fidelity.q_reason = True

# ── Faithfulness sub-toggles ────────────────────────────
_C.faithfulness = CN()
_C.faithfulness.val_step = True
_C.faithfulness.align_entail = True

# ── Robustness ───────────────────────────────────────────
_C.robustness = CN()
_C.robustness.in_domain_accuracy = 0.0
_C.robustness.out_domain_accuracy = 0.0


def get_cfg(config_file_path):
    config = _C.clone()
    if config_file_path:
        config.merge_from_file(config_file_path)
    if config.output_dir and not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)
    return config
