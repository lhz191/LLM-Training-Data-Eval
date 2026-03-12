"""
Robustness — Out-of-Distribution Accuracy Gap

Survey Trustworthy / Robustness:

    Δ_OOD = Acc_RM^{in} - Acc_RM^{out}

Compares reward model (or solver) accuracy on in-domain vs out-of-domain splits.
A smaller gap indicates more stable preference ranking across domains.

Note: The FAIRR consistency metric C(T,s) is now in validity.py
(Survey Section 3.2 — "robustness style validity checks").
"""

from typing import Dict


def compute_ood_gap(
    in_domain_accuracy: float,
    out_domain_accuracy: float,
) -> Dict[str, float]:
    """Δ_OOD = Acc_in - Acc_out."""
    return {
        "in_domain_accuracy": in_domain_accuracy,
        "out_domain_accuracy": out_domain_accuracy,
        "ood_gap": in_domain_accuracy - out_domain_accuracy,
    }
