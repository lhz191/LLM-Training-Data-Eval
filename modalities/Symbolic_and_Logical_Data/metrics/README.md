# Symbolic & Logical Data — Layer 1 Metrics

Modality-level metrics for mathematical reasoning, formal logic, and code data.

## Metrics

| Metric | File | What it measures | Reference |
|--------|------|-----------------|-----------|
| **Val_step** | [faithfulness.py](faithfulness.py) | Fraction of valid reasoning steps | ReCEval (Prasad et al., 2023); FRODO (Paul et al., 2024) |
| **Align_entail** | [faithfulness.py](faithfulness.py) | Entailment alignment between consecutive reasoning steps | Prasad et al., 2023; Yao & Barbosa, 2024 |
| **Acc_SC** | [fidelity.py](fidelity.py) | Self-consistency accuracy via majority vote | Wang et al., 2023a |
| **Agree** | [fidelity.py](fidelity.py) | Pairwise answer agreement across K sampled chains | — |
| **rho_LLM-human** | [fidelity.py](fidelity.py) | Spearman rank correlation between LLM and human judgments | JudgeLM (Zhu et al., 2025); MT-Bench (Zheng et al., 2023) |
| **Q_reason** | [fidelity.py](fidelity.py) | Aggregated rubric score over M evaluation dimensions | — |
| **Acc_verify** | [validity.py](validity.py) | Verification accuracy: solution correctness checked by external verifier | MetaMath (Yu et al., 2024a); OpenMathInstruct-1 (Toshniwal et al., 2024) |
| **PassRate** | [validity.py](validity.py) | Unit-test pass rate for code solutions | OpenCodeInstruct (Ahmad et al., 2025) |
| **Acc_proof** | [validity.py](validity.py) | Joint accuracy of proof label and reasoning chain | ProofWriter (Tafjord et al., 2021) |
| **C(T,s)** | [validity.py](validity.py) | Consistency under input perturbation (rule-set shuffling) | FAIRR (Sanyal et al., 2022) |
| **Delta_OOD** | [robustness.py](robustness.py) | Out-of-distribution accuracy gap between in-domain and out-domain | — |

## Relationship to Layer 2

See [math_eval/metrics/](../math_eval/metrics/) for task-specific metrics: format_check, validity (code execution), faithfulness, reasoning_validity (LLM judge).
