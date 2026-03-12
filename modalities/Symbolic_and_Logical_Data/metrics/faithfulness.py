"""
Faithfulness — Step Validity & Entailment Alignment

Survey Section 3.3 (Trustworthy — Faithfulness):

  f_exec(c_{i,t}) = 1 if c_{i,t} is a valid transformation, else 0

  Val_step = (1 / Σᵢ Tᵢ) Σᵢ Σₜ f_exec(c_{i,t})
      Ref: ReCEval (Prasad et al., 2023), FRODO (Paul et al., 2024)

  p_{i,t} = g_entail(c_{i,t-1} ⇒ c_{i,t})

  Align_entail = (1 / Σᵢ(Tᵢ − 1)) Σᵢ Σ_{t=2}^{Tᵢ} p_{i,t}
      Ref: Prasad et al., 2023; Yao & Barbosa, 2024
"""

from typing import List, Dict, Any, Callable


# ═══════════════════════════════════════════════════════════════
# Val_step — Step Validity Rate
# ═══════════════════════════════════════════════════════════════

def compute_val_step(
    data: List[Dict[str, Any]],
    f_exec: Callable[[str], bool],
) -> Dict[str, Any]:
    """
    Val_step = (1 / Σᵢ Tᵢ) Σᵢ Σₜ f_exec(c_{i,t})

    Args:
        data: list of reasoning examples, each with:
              "steps": list[str]  — chain-of-thought steps [c_{i,1}, ..., c_{i,T}]
        f_exec: f_exec(step) -> bool
                Domain-specific step verifier.
                e.g. symbolic executor, code compiler, entailment checker.
    """
    total_steps = 0
    valid_steps = 0

    for item in data:
        steps = item.get("steps", [])
        for step in steps:
            total_steps += 1
            if f_exec(step):
                valid_steps += 1

    rate = valid_steps / total_steps if total_steps > 0 else 0.0
    return {
        "val_step": rate,
        "valid_steps": valid_steps,
        "total_steps": total_steps,
    }


# ═══════════════════════════════════════════════════════════════
# Align_entail — Entailment Alignment
# ═══════════════════════════════════════════════════════════════

def compute_align_entail(
    data: List[Dict[str, Any]],
    g_entail: Callable[[str, str], float],
) -> Dict[str, Any]:
    """
    Align_entail = (1 / Σᵢ(Tᵢ − 1)) Σᵢ Σ_{t=2}^{Tᵢ} p_{i,t}

    p_{i,t} = g_entail(c_{i,t-1} ⇒ c_{i,t})

    Args:
        data: list of reasoning examples, each with:
              "steps": list[str]  — chain-of-thought steps [c_{i,1}, ..., c_{i,T}]
        g_entail: g_entail(premise, hypothesis) -> float
                  Entailment scorer returning p ∈ [0, 1].
                  e.g. NLI cross-encoder, textual entailment model.
    """
    total_pairs = 0
    score_sum = 0.0

    for item in data:
        steps = item.get("steps", [])
        for t in range(1, len(steps)):
            total_pairs += 1
            score_sum += g_entail(steps[t - 1], steps[t])

    alignment = score_sum / total_pairs if total_pairs > 0 else 0.0
    return {
        "align_entail": alignment,
        "total_pairs": total_pairs,
    }
