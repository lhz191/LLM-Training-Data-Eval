"""
Validity — Quality Metrics for Symbolic & Logical Data

Survey Section 3.2:

  Acc_verify = (1/N) Σ f_check(qᵢ, cᵢ, yᵢ)
      Ref: MetaMath (Yu et al., 2024a), OpenMathInstruct-1 (Toshniwal et al., 2024)

  PassRateᵢ = (1/|Tᵢ|) Σ_{t∈Tᵢ} 𝟙( exec(yᵢ, t) = pass )
  PassRate   = (1/N) Σ PassRateᵢ
      Ref: OpenCodeInstruct (Ahmad et al., 2025)

  Acc_proof = (1/N) Σ 𝟙(yᵢ = y⋆ᵢ) · 𝟙(cᵢ = c⋆ᵢ)
      Ref: ProofWriter (Tafjord et al., 2021)

  C(T,s) = (1/K) Σ_{k=1}^{K} 𝟙( f(T,s) = f(T'ₖ, s'ₖ) )
      Ref: FAIRR (Sanyal et al., 2022) — eval_consistency.py
           https://github.com/INK-USC/FaiRR
"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════
# Acc_verify — Verification Accuracy
# ═══════════════════════════════════════════════════════════════

def compute_acc_verify(
    data: List[Dict[str, Any]],
    f_check: Callable[[str, str, str], bool],
) -> Dict[str, Any]:
    """
    Acc_verify = (1/N) Σ f_check(qᵢ, cᵢ, yᵢ)

    Args:
        data: list of reasoning examples, each with:
              "question": str (qᵢ — problem)
              "rationale": str (cᵢ — chain-of-thought / code trace)
              "answer": str (yᵢ — final answer)
        f_check: f_check(q, c, y) -> bool
                 Domain-specific verification checker.
    """
    correct = 0
    total = len(data)

    for item in data:
        q = item.get("question", "")
        c = item.get("rationale", "")
        y = item.get("answer", "")
        if f_check(q, c, y):
            correct += 1

    acc = correct / total if total > 0 else 0.0
    return {"acc_verify": acc, "correct": correct, "total": total}


# ═══════════════════════════════════════════════════════════════
# PassRate — Unit-Test Pass Rate
# Ref: OpenCodeInstruct (Ahmad et al., 2025)
# ═══════════════════════════════════════════════════════════════

def compute_pass_rate(
    data: List[Dict[str, Any]],
    exec_fn: Optional[Callable[[str, str], bool]] = None,
) -> Dict[str, Any]:
    """
    PassRateᵢ = (1/|Tᵢ|) Σ 𝟙( exec(yᵢ, t) = pass )
    PassRate   = (1/N) Σ PassRateᵢ

    Args:
        data: list of dicts, each with:
              "solution": str — generated code yᵢ
              "tests": list[str] — test suite Tᵢ
        exec_fn: exec_fn(solution, test) -> bool (pass/fail)
                 If None, expects pre-computed "test_results": list[bool]
    """
    per_sample: List[float] = []

    for item in data:
        if exec_fn is not None:
            solution = item.get("solution", "")
            tests = item.get("tests", [])
            if not tests:
                continue
            passed = sum(exec_fn(solution, t) for t in tests)
            per_sample.append(passed / len(tests))
        else:
            results = item.get("test_results", [])
            if not results:
                continue
            per_sample.append(sum(results) / len(results))

    dataset_rate = float(np.mean(per_sample)) if per_sample else 0.0
    return {"pass_rate": dataset_rate, "num_samples": len(per_sample)}


# ═══════════════════════════════════════════════════════════════
# Acc_proof — Strict Proof Accuracy
# Ref: ProofWriter (Tafjord et al., 2021)
# ═══════════════════════════════════════════════════════════════

def _normalize_proof(proof) -> str:
    """Normalize a proof representation for exact-match comparison."""
    if isinstance(proof, list):
        return str(sorted(str(p).strip().lower() for p in proof))
    return str(proof).strip().lower()


def compute_acc_proof(
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Acc_proof = (1/N) Σ 𝟙(yᵢ = y⋆ᵢ) · 𝟙(cᵢ = c⋆ᵢ)

    Both the entailment label AND the proof graph must exactly match gold.

    Expected fields:
        "predicted_label" / "gold_label"   — answer label y / y⋆
        "predicted_proof" / "gold_proof"   — proof graph c / c⋆
    """
    correct = 0
    total = 0

    for item in data:
        gold_label = item.get("gold_label")
        gold_proof = item.get("gold_proof")
        if gold_label is None or gold_proof is None:
            continue
        total += 1

        pred_label = item.get("predicted_label")
        pred_proof = item.get("predicted_proof")

        label_match = str(pred_label).strip().lower() == str(gold_label).strip().lower()
        proof_match = _normalize_proof(pred_proof) == _normalize_proof(gold_proof)

        if label_match and proof_match:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    return {"acc_proof": acc, "correct": correct, "total": total}


# ═══════════════════════════════════════════════════════════════
# C(T,s) — FAIRR Consistency
# Ref: FAIRR (Sanyal et al., 2022) — eval_consistency.py
#      https://github.com/INK-USC/FaiRR
# ═══════════════════════════════════════════════════════════════

def compute_consistency(
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    C(T,s) = (1/K) Σ_{k=1}^{K} 𝟙( f(T,s) = f(T'ₖ, s'ₖ) )

    Following FAIRR eval_consistency.py:
    group by equivalence set, compare original prediction with perturbed ones.

    Expected fields:
        "group_id": str/int — equivalence-set identifier
        "is_original": bool — True for original (T,s), False for perturbation
        "prediction": str — model's predicted label f(·)
    """
    groups: Dict[Any, Dict[str, Any]] = defaultdict(
        lambda: {"original": None, "perturbed": []}
    )

    for item in data:
        gid = item.get("group_id")
        if gid is None:
            continue
        pred = str(item.get("prediction", "")).strip()
        if item.get("is_original", False):
            groups[gid]["original"] = pred
        else:
            groups[gid]["perturbed"].append(pred)

    consistencies: List[float] = []
    for grp in groups.values():
        orig = grp["original"]
        perturbed = grp["perturbed"]
        if orig is None or not perturbed:
            continue
        matches = sum(1 for p in perturbed if p == orig)
        consistencies.append(matches / len(perturbed))

    avg = float(np.mean(consistencies)) if consistencies else 0.0
    return {"consistency": avg, "num_groups": len(consistencies)}
