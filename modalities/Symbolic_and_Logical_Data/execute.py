"""
Symbolic & Logical Data Evaluation Runner

Loads data via math_eval loaders, uses math_eval executors
(CodeExecutor, compare_math_answers) as domain checkers,
then computes survey-aligned metrics.

Quality:
  Validity    — Acc_verify, PassRate, Acc_proof, C(T,s)
  Fidelity    — Acc_SC, Agree, ρ_LLM-human, Q_reason
Trustworthy:
  Faithfulness — Val_step, Align_entail
  Robustness   — OOD Gap

Usage:
    python execute.py -f configs/test.yaml
"""

import argparse
import os
import sys
import json

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MATH_EVAL_DIR = os.path.join(_THIS_DIR, "math_eval")
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _MATH_EVAL_DIR not in sys.path:
    sys.path.append(_MATH_EVAL_DIR)

from configs.basic_cfg import get_cfg

from metrics.validity import (
    compute_acc_verify,
    compute_pass_rate,
    compute_acc_proof,
    compute_consistency,
)
from metrics.fidelity import (
    compute_acc_sc,
    compute_agreement,
    compute_llm_human_correlation,
    compute_q_reason,
)
from metrics.faithfulness import (
    compute_val_step,
    compute_align_entail,
)
from metrics.robustness import (
    compute_ood_gap,
)


# ────────────────────────────────────────────────────────────
# Data loading: math_eval loaders + executors
# ────────────────────────────────────────────────────────────

def _get_loader(dataset_name, data_path):
    """Get the appropriate loader from math_eval."""
    from loaders import (
        MetaMathQALoader,
        OpenMathInstructLoader,
        GSM8KAugLoader,
        LILALoader,
    )

    name = dataset_name.lower().replace("_", "").replace("-", "").replace(" ", "")
    loaders = {
        "openmath": OpenMathInstructLoader,
        "openmathinstruct": OpenMathInstructLoader,
        "openmathinstruct1": OpenMathInstructLoader,
        "metamath": MetaMathQALoader,
        "metamathqa": MetaMathQALoader,
        "gsm8kaug": GSM8KAugLoader,
        "lila": LILALoader,
    }
    loader_cls = loaders.get(name)
    if loader_cls is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(loaders.keys())}"
        )
    return loader_cls(data_path)


def _build_f_check(executor_type):
    """
    Build f_check(question, rationale, answer) -> bool
    using math_eval's code executor + answer comparator.

    For code-based solutions (openmath/lila): extract code → execute → compare.
    Falls back to compare_math_answers on the answer string directly.
    """
    from code_executor import (
        get_executor,
        get_code_extractor,
        get_answer_extractor,
        compare_math_answers,
    )
    import executor as _  # noqa: trigger registration

    try:
        code_exec = get_executor(executor_type)
    except (ValueError, KeyError):
        code_exec = None

    try:
        code_extractor = get_code_extractor(executor_type)
    except (ValueError, KeyError):
        code_extractor = None

    try:
        answer_extractor = get_answer_extractor(executor_type)
    except (ValueError, KeyError):
        answer_extractor = None

    def f_check(question: str, rationale: str, answer: str) -> bool:
        if code_exec is not None and code_extractor is not None:
            code = code_extractor.extract(rationale)
            if code:
                result, error = code_exec.execute(code)
                if result is not None and not error:
                    return compare_math_answers(str(result), str(answer))

        predicted = answer
        if answer_extractor is not None:
            try:
                predicted = answer_extractor.extract(rationale) or answer
            except Exception:
                predicted = answer

        return compare_math_answers(str(predicted), str(answer))

    return f_check


def _build_exec_fn(executor_type):
    """
    Build exec_fn(solution, test) -> bool
    using math_eval's code executor.
    """
    from code_executor import get_executor
    import executor as _  # noqa

    try:
        code_exec = get_executor(executor_type)
    except (ValueError, KeyError):
        return None

    def exec_fn(solution: str, test: str) -> bool:
        full_code = solution + "\n" + test
        result, error = code_exec.execute(full_code)
        return error is None or error == ""

    return exec_fn


def _build_f_exec(executor_type):
    """
    Build f_exec(step) -> bool for step-level validity.

    Tries to execute each step as code; if it's not code,
    returns True (natural language steps are assumed valid
    unless a domain-specific checker says otherwise).
    """
    from code_executor import get_executor
    import executor as _  # noqa

    try:
        code_exec = get_executor(executor_type)
    except (ValueError, KeyError):
        return None

    def f_exec(step: str) -> bool:
        if not any(kw in step for kw in ["=", "print", "import", "def ", "return"]):
            return True
        try:
            result, error = code_exec.execute(step)
            return error is None or error == ""
        except Exception:
            return True

    return f_exec


def audit_samples(samples):
    """
    Data-level quality audit: extract intrinsic properties from each
    MathSample into a unified record for metric computation.
    """
    records = []
    for s in samples:
        solution = s.solution
        if isinstance(solution, list):
            solution = "\n".join(solution)

        records.append({
            "sample_id": s.sample_id,
            "question": s.question,
            "rationale": solution,
            "answer": str(s.ground_truth) if s.ground_truth is not None else "",
            "reference": str(s.ground_truth) if s.ground_truth is not None else "",
            "steps": [step.strip() for step in solution.split("\n") if step.strip()],
        })
    return records


# ────────────────────────────────────────────────────────────
# Legacy JSONL loader
# ────────────────────────────────────────────────────────────

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON parse error at line {line_num}:\n{line}"
                ) from e
    return data


# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────

def main(args):
    config = get_cfg(args.config_file)

    # ── Load data ────────────────────────────────────────────
    data = []
    executor_type = None

    if config.dataset and config.data_path:
        print(f"[Loader] dataset={config.dataset}, path={config.data_path}")
        loader = _get_loader(config.dataset, config.data_path)
        samples = list(loader.iterate())
        print(f"[Loader] Loaded {len(samples)} samples")

        executor_type = config.executor_type or config.dataset
        data = audit_samples(samples)
        print(f"[Audit] {len(data)} data points prepared")

    elif config.jsonl_path and os.path.isfile(config.jsonl_path):
        print(f"[Legacy] Loading raw JSONL: {config.jsonl_path}")
        data = load_jsonl(config.jsonl_path)

    else:
        print("[Warning] No data_path or jsonl_path configured")

    outputs = {}

    # ── Quality: Validity ───────────────────────────────────
    if config.metrics.validity and data:
        vc = config.validity

        if vc.acc_verify:
            if executor_type:
                f_check = _build_f_check(executor_type)
                print("[Validity] Acc_verify ...")
                outputs["acc_verify"] = compute_acc_verify(data, f_check=f_check)
            else:
                print("[Validity] Acc_verify skipped — no executor configured")

        if vc.pass_rate:
            exec_fn = _build_exec_fn(executor_type) if executor_type else None
            print("[Validity] PassRate ...")
            outputs["pass_rate"] = compute_pass_rate(data, exec_fn=exec_fn)

        if vc.acc_proof:
            print("[Validity] Acc_proof ...")
            outputs["acc_proof"] = compute_acc_proof(data)

        if vc.consistency:
            print("[Validity] Consistency C(T,s) ...")
            outputs["consistency"] = compute_consistency(data)

    # ── Quality: Fidelity ───────────────────────────────────
    if config.metrics.fidelity and data:
        fc = config.fidelity

        if fc.acc_sc:
            print("[Fidelity] Acc_SC ...")
            outputs["acc_sc"] = compute_acc_sc(data)

        if fc.agreement:
            print("[Fidelity] Agree ...")
            outputs["agreement"] = compute_agreement(data)

        if fc.llm_human_correlation:
            print("[Fidelity] ρ_LLM-human ...")
            outputs["llm_human_correlation"] = compute_llm_human_correlation(data)

        if fc.q_reason:
            print("[Fidelity] Q_reason ...")
            outputs["q_reason"] = compute_q_reason(data)

    # ── Trustworthy: Faithfulness ───────────────────────────
    if config.metrics.faithfulness and data:
        fh = config.faithfulness

        if fh.val_step:
            f_exec = _build_f_exec(executor_type) if executor_type else None
            if f_exec is not None:
                print("[Faithfulness] Val_step ...")
                outputs["val_step"] = compute_val_step(data, f_exec=f_exec)
            else:
                print("[Faithfulness] Val_step skipped — no executor")

        if fh.align_entail:
            print("[Faithfulness] Align_entail skipped — g_entail not provided")

    # ── Trustworthy: Robustness ─────────────────────────────
    if config.metrics.robustness:
        rc = config.robustness
        print("[Robustness] OOD Gap ...")
        outputs["ood_gap"] = compute_ood_gap(
            in_domain_accuracy=rc.in_domain_accuracy,
            out_domain_accuracy=rc.out_domain_accuracy,
        )

    # ── Save ────────────────────────────────────────────────
    output_path = os.path.join(config.output_dir or ".", "res.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file", type=str, default="")
    args = parser.parse_args()
    main(args)
