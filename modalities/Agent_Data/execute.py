"""
Agent Data Evaluation Runner

Loads data via api_agent_eval loaders, runs api_agent_eval executors
as checkers, then computes survey-aligned metrics.

Quality:
  Validity  — ExecRate, SRvalid, PC
  Fidelity  — FID, FVD, SSIM, MSE/PSNR, LPIPS, SemAlign
  Diversity — AD, RD, VBench

Trustworthy:
  Safety    — RVR, RVR(t), RI, MSCR, Kinematics, SafetySat,
              Hazard Rejection/Risk, TTC, MDC

Usage:
    python execute.py -f configs/test.yaml
"""

import argparse
import os
import sys
import json
import re

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_API_EVAL_DIR = os.path.join(_THIS_DIR, "api_agent_eval")
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _API_EVAL_DIR not in sys.path:
    sys.path.append(_API_EVAL_DIR)

from configs.basic_cfg import get_cfg

from metrics.validity import (
    compute_exec_rate,
    compute_success_rate,
    compute_percent_complete,
)
from metrics.fidelity import (
    compute_fid,
    compute_fvd,
    compute_ssim,
    compute_mse_psnr,
    compute_lpips,
    compute_semantic_alignment,
)
from metrics.diversity import (
    compute_agent_diversity,
    compute_road_diversity,
    compute_vbench,
)
from metrics.safety import (
    compute_rvr,
    compute_rvr_by_type,
    compute_route_incompleteness,
    compute_mscr,
    compute_kinematics,
    compute_safety_satisfaction,
    compute_hazard_rejection,
    compute_ttc,
    compute_mdc,
)


# ────────────────────────────────────────────────────────────
# Data loading: api_agent_eval loaders + executors
# ────────────────────────────────────────────────────────────

def _get_loader(dataset_name, data_path):
    """Get the appropriate loader from api_agent_eval."""
    from loaders import ToolBenchLoader, XLAMLoader

    name = dataset_name.lower().replace("_", "-").replace(" ", "")
    if name == "toolbench":
        return ToolBenchLoader(data_path)
    elif name in ("xlam", "xlam-60k"):
        return XLAMLoader(data_path)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: toolbench, xlam"
        )


def _get_checker(executor_type, **kwargs):
    """Get the appropriate ExecutabilityChecker from api_agent_eval."""
    from api_executor import get_executability_checker
    import executor as _  # noqa: trigger auto-registration
    return get_executability_checker(executor_type, **kwargs)


def _checker_errors_to_failed_indices(errors):
    """Parse checker error strings to find which api_call indices failed."""
    failed = set()
    for err in errors:
        m = re.search(r"api_calls\[(\d+)\]", err)
        if m:
            failed.add(int(m.group(1)))
    return failed


def audit_samples(samples, checker=None):
    """
    Data-level quality audit: evaluate intrinsic properties of each
    data point using the domain executor/checker.

    For each APIAgentSample, runs the ExecutabilityChecker and produces
    a per-sample quality record with per-action executability verdicts.

    Returns List[Dict] — one audit record per data point.
    """
    records = []

    for sample in samples:
        failed_indices = set()
        if checker is not None:
            errors, _warnings, _stats = checker.check(sample)
            failed_indices = _checker_errors_to_failed_indices(errors)

        actions = []
        for i, call in enumerate(sample.api_calls):
            actions.append({
                "name": call.name,
                "arguments": call.arguments,
                "executed": i not in failed_indices,
            })

        record = {
            "sample_id": sample.sample_id,
            "query": sample.query,
            "actions": actions,
            "states": [None] * len(actions),
            "tools": [t.name for t in sample.tools],
            "success": len(failed_indices) == 0,
        }
        records.append(record)

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
# Image helpers (for fidelity metrics)
# ────────────────────────────────────────────────────────────

def _load_image_pairs(data):
    from PIL import Image

    real_imgs, gen_imgs = [], []
    for entry in data:
        rp = entry.get("real_image_path")
        gp = entry.get("gen_image_path")
        if rp and gp and os.path.isfile(rp) and os.path.isfile(gp):
            real_imgs.append(np.array(Image.open(rp).convert("RGB")))
            gen_imgs.append(np.array(Image.open(gp).convert("RGB")))
    return real_imgs, gen_imgs


def _load_gen_images(data):
    from PIL import Image

    images = []
    for entry in data:
        ip = entry.get("gen_image_path", "")
        if ip and os.path.isfile(ip):
            images.append(np.array(Image.open(ip).convert("RGB")))
    return images


# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────

def main(args):
    config = get_cfg(args.config_file)

    # ── Load data ────────────────────────────────────────────
    data = []

    if config.dataset and config.data_path:
        print(f"[Loader] dataset={config.dataset}, path={config.data_path}")
        loader = _get_loader(config.dataset, config.data_path)
        samples = list(loader.iterate())
        print(f"[Loader] Loaded {len(samples)} samples")

        checker = None
        executor_type = config.executor_type or config.dataset
        try:
            checker_kwargs = {}
            if config.toolenv_path and executor_type.lower() == "toolbench":
                checker_kwargs["toolenv_path"] = config.toolenv_path
            checker = _get_checker(executor_type, **checker_kwargs)
            print(f"[Checker] Using {executor_type} ExecutabilityChecker")
        except (ValueError, ImportError, TypeError) as e:
            print(f"[Checker] No checker for '{executor_type}': {e}")

        data = audit_samples(samples, checker=checker)
        print(f"[Audit] {len(data)} data points audited")

    elif config.jsonl_path and os.path.isfile(config.jsonl_path):
        print(f"[Legacy] Loading raw JSONL: {config.jsonl_path}")
        data = load_jsonl(config.jsonl_path)
    else:
        print("[Warning] No data_path or jsonl_path configured")

    outputs = {}

    # ── Quality: Validity ───────────────────────────────────
    if config.metrics.validity and data:
        print("[Validity] ExecRate ...")
        outputs["exec_rate"] = compute_exec_rate(data)

        print("[Validity] SRvalid ...")
        outputs["success_rate"] = compute_success_rate(data)

        print("[Validity] PercentComplete ...")
        outputs["percent_complete"] = compute_percent_complete(data)

    # ── Quality: Fidelity ───────────────────────────────────
    if config.metrics.fidelity and data:
        fc = config.fidelity

        if fc.fid.enabled:
            print("[Fidelity] FID ...")
            outputs["fid"] = compute_fid(
                fc.fid.real_dir,
                fc.fid.gen_dir,
                mode=fc.fid.mode,
                batch_size=fc.fid.batch_size,
            )

        if fc.fvd.enabled:
            print("[Fidelity] FVD ...")
            feats_real = np.load(fc.fvd.real_features_path)
            feats_gen = np.load(fc.fvd.gen_features_path)
            outputs["fvd"] = compute_fvd(feats_real, feats_gen)

        need_pairs = fc.ssim or fc.mse_psnr or fc.lpips.enabled
        real_imgs, gen_imgs = (
            _load_image_pairs(data) if need_pairs else ([], [])
        )

        if fc.ssim and real_imgs:
            print("[Fidelity] SSIM ...")
            outputs["ssim"] = compute_ssim(real_imgs, gen_imgs)

        if fc.mse_psnr and real_imgs:
            print("[Fidelity] MSE / PSNR ...")
            outputs["mse_psnr"] = compute_mse_psnr(real_imgs, gen_imgs)

        if fc.lpips.enabled and real_imgs:
            print("[Fidelity] LPIPS ...")
            outputs["lpips"] = compute_lpips(
                real_imgs, gen_imgs, net=fc.lpips.net
            )

        if fc.semantic_alignment.enabled:
            print("[Fidelity] SemAlign / SemFid / MatchRate ...")
            prompts = [e["prompt"] for e in data if "prompt" in e]
            images = _load_gen_images(data)
            if prompts and images and len(prompts) == len(images):
                outputs["semantic_alignment"] = compute_semantic_alignment(
                    prompts,
                    images,
                    model_name=fc.semantic_alignment.model_name,
                    threshold=fc.semantic_alignment.threshold,
                    batch_size=fc.semantic_alignment.batch_size,
                )

    # ── Quality: Diversity ──────────────────────────────────
    if config.metrics.diversity and data:
        dc = config.diversity

        if dc.agent_diversity:
            print("[Diversity] Agent Diversity (AD) ...")
            outputs["agent_diversity"] = compute_agent_diversity(data)

        if dc.road_diversity:
            print("[Diversity] Road Diversity (RD) ...")
            outputs["road_diversity"] = compute_road_diversity(data)

        if dc.vbench.enabled:
            print("[Diversity] VBench ...")
            video_paths = [e["video_path"] for e in data if "video_path" in e]
            if video_paths:
                dims = (
                    list(dc.vbench.dimensions)
                    if dc.vbench.dimensions
                    else None
                )
                outputs["vbench"] = compute_vbench(
                    video_paths,
                    full_info_dir=dc.vbench.full_json_dir,
                    output_path=dc.vbench.output_path,
                    dimensions=dims,
                )

    # ── Trustworthy: Safety ─────────────────────────────────
    if config.metrics.safety and data:
        sc = config.safety
        kc = config.kinematics

        if sc.rvr:
            print("[Safety] RVR ...")
            outputs["rvr"] = compute_rvr(data)

        if sc.rvr_by_type:
            print("[Safety] RVR by type ...")
            outputs["rvr_by_type"] = compute_rvr_by_type(data)

        if sc.route_incompleteness:
            print("[Safety] Route Incompleteness (RI) ...")
            outputs["route_incompleteness"] = compute_route_incompleteness(
                data
            )

        if sc.mscr:
            print("[Safety] MSCR ...")
            outputs["mscr"] = compute_mscr(data)

        if sc.kinematics:
            print("[Safety] Kinematics (ACC / YV / JerkRMS / HardBrake) ...")
            outputs["kinematics"] = compute_kinematics(
                data,
                dt=kc.dt,
                hard_brake_threshold=kc.hard_brake_threshold,
            )

        if sc.safety_satisfaction:
            print("[Safety] SafetySat ...")
            outputs["safety_satisfaction"] = compute_safety_satisfaction(data)

        if sc.hazard_rejection:
            print("[Safety] Hazard Rejection / Risk ...")
            outputs["hazard_rejection"] = compute_hazard_rejection(data)

        if sc.ttc:
            print("[Safety] TTC ...")
            outputs["ttc"] = compute_ttc(data)

        if sc.mdc:
            print("[Safety] MDC ...")
            outputs["mdc"] = compute_mdc(data)

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
