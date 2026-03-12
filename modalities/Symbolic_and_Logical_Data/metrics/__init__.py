"""
Symbolic & Logical Data Metrics — Survey Sections 3.2 & 3.3

Quality:
  Validity    — Acc_verify, PassRate, Acc_proof, C(T,s)
  Fidelity    — Acc_SC, Agree, ρ_LLM-human, Q_reason

Trustworthy:
  Faithfulness — Val_step, Align_entail
  Robustness   — OOD Gap
"""

from .validity import (
    compute_acc_verify,
    compute_pass_rate,
    compute_acc_proof,
    compute_consistency,
)
from .fidelity import (
    compute_acc_sc,
    compute_agreement,
    compute_llm_human_correlation,
    compute_q_reason,
)
from .faithfulness import (
    compute_val_step,
    compute_align_entail,
)
from .robustness import (
    compute_ood_gap,
)

__all__ = [
    # Validity
    "compute_acc_verify",
    "compute_pass_rate",
    "compute_acc_proof",
    "compute_consistency",
    # Fidelity
    "compute_acc_sc",
    "compute_agreement",
    "compute_llm_human_correlation",
    "compute_q_reason",
    # Faithfulness
    "compute_val_step",
    "compute_align_entail",
    # Robustness
    "compute_ood_gap",
]
