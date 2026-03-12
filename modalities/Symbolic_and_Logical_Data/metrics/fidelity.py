"""
Fidelity — Proxy-Signal Reliability for Symbolic & Logical Data

Survey Section 3.2 (Fidelity):

  Acc_SC = (1/N) Σ 𝟙(ŷᵢ = yᵢ)
      ŷᵢ = argmax_y Σ_k 𝟙(y_{i,k} = y)     (majority vote)
      Ref: Self-Consistency (Wang et al., 2023a)

  Agree(qᵢ) = (1 / K(K−1)) Σ_{k≠k'} 𝟙(y_{i,k} = y_{i,k'})
      pairwise answer agreement across K sampled chains

  ρ_LLM-human = corrSpearman( u^{LLM}, u^{human} )
      Ref: JudgeLM, MT-Bench (Zhu et al., 2025; Zheng et al., 2023)

  Q_reason = (1/M) Σ_{m=1}^{M} score_{i,m}
      aggregated rubric score over M dimensions

Libraries:
    scipy.stats.spearmanr   — for ρ_LLM-human
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter


# ═══════════════════════════════════════════════════════════════
# Acc_SC — Self-Consistency Accuracy
# Ref: Wang et al., 2023a
# ═══════════════════════════════════════════════════════════════

def compute_acc_sc(
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Acc_SC = (1/N) Σ 𝟙(ŷᵢ = yᵢ)

    ŷᵢ = argmax_y Σ_k 𝟙(y_{i,k} = y)   (majority vote over K samples)

    Expected fields per question:
        "sampled_answers": list[str]  — K answers {y_{i,k}}
        "reference": str              — ground-truth yᵢ
    """
    correct = 0
    total = 0

    for item in data:
        answers = item.get("sampled_answers", [])
        ref = item.get("reference")
        if not answers or ref is None:
            continue
        total += 1

        majority = Counter(answers).most_common(1)[0][0]
        if str(majority).strip() == str(ref).strip():
            correct += 1

    acc = correct / total if total > 0 else 0.0
    return {"acc_sc": acc, "correct": correct, "total": total}


# ═══════════════════════════════════════════════════════════════
# Agree — Pairwise Answer Agreement
# ═══════════════════════════════════════════════════════════════

def compute_agreement(
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Agree(qᵢ) = (1 / K(K−1)) Σ_{k≠k'} 𝟙(y_{i,k} = y_{i,k'})

    Averaged over the dataset.

    Expected fields per question:
        "sampled_answers": list[str]  — K answers {y_{i,k}}
    """
    agreements: List[float] = []

    for item in data:
        answers = item.get("sampled_answers", [])
        K = len(answers)
        if K < 2:
            continue

        pairs_match = 0
        total_pairs = K * (K - 1)
        for k in range(K):
            for kp in range(K):
                if k != kp and answers[k] == answers[kp]:
                    pairs_match += 1

        agreements.append(pairs_match / total_pairs)

    avg = float(np.mean(agreements)) if agreements else 0.0
    return {"agreement_mean": avg, "num_questions": len(agreements)}


# ═══════════════════════════════════════════════════════════════
# ρ_LLM-human — Spearman Rank Correlation
# Ref: JudgeLM (Zhu et al., 2025), MT-Bench (Zheng et al., 2023)
# ═══════════════════════════════════════════════════════════════

def compute_llm_human_correlation(
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    ρ_LLM-human = corrSpearman( u^{LLM}, u^{human} )

    Expected fields per example:
        "llm_score": float   — u^{LLM}_i
        "human_score": float — u^{human}_i
    """
    from scipy.stats import spearmanr, pearsonr

    llm_scores = []
    human_scores = []

    for item in data:
        ls = item.get("llm_score")
        hs = item.get("human_score")
        if ls is not None and hs is not None:
            llm_scores.append(float(ls))
            human_scores.append(float(hs))

    if len(llm_scores) < 2:
        return {"spearman_rho": None, "pearson_r": None, "num_pairs": len(llm_scores)}

    rho, p_spearman = spearmanr(llm_scores, human_scores)
    r, p_pearson = pearsonr(llm_scores, human_scores)

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_spearman),
        "pearson_r": float(r),
        "pearson_p": float(p_pearson),
        "num_pairs": len(llm_scores),
    }


# ═══════════════════════════════════════════════════════════════
# Q_reason — Aggregated Rubric Score
# ═══════════════════════════════════════════════════════════════

def compute_q_reason(
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Q_reason = (1/M) Σ_{m=1}^{M} score_{i,m}

    Averaged over the dataset.

    Expected fields per example:
        "rubric_scores": dict {dimension_name: float}
    """
    per_sample: List[float] = []

    for item in data:
        scores = item.get("rubric_scores", {})
        if not scores:
            continue
        per_sample.append(sum(scores.values()) / len(scores))

    avg = float(np.mean(per_sample)) if per_sample else 0.0
    return {"q_reason_mean": avg, "num_samples": len(per_sample)}


