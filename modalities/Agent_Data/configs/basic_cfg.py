"""
Config for Agent Data Evaluation

Data loading:
  dataset   — toolbench / xlam / arcee-agent / general
  data_path — path to dataset file

Quality:
  Validity  — ExecRate, SRvalid, PC
  Fidelity  — FID, FVD, SSIM, MSE/PSNR, LPIPS, SemAlign
  Diversity — AD, RD, VBench

Trustworthy:
  Safety    — RVR, RVR(t), RI, MSCR, Kinematics, SafetySat,
              Hazard Rejection/Risk, TTC, MDC
"""

import os
from yacs.config import CfgNode as CN

_C = CN()

# ══════════════════════════════════════════════════════════
# Data loading — uses api_agent_eval loaders + executors
# ══════════════════════════════════════════════════════════
_C.dataset = ""            # toolbench / xlam / arcee-agent / general
_C.data_path = ""          # path to dataset file
_C.executor_type = ""      # executor/checker to use (defaults to same as dataset)
_C.toolenv_path = ""       # ToolBench-specific: path to toolenv/tools

_C.jsonl_path = ""         # legacy: raw JSONL (bypasses loader/executor)
_C.output_dir = "eval_res"

# ══════════════════════════════════════════════════════════
# Metric-family toggles
# ══════════════════════════════════════════════════════════
_C.metrics = CN()
_C.metrics.validity = True
_C.metrics.fidelity = False
_C.metrics.diversity = True
_C.metrics.safety = True

# ══════════════════════════════════════════════════════════
# Fidelity sub-config
# ══════════════════════════════════════════════════════════
_C.fidelity = CN()

_C.fidelity.fid = CN()
_C.fidelity.fid.enabled = False
_C.fidelity.fid.real_dir = ""
_C.fidelity.fid.gen_dir = ""
_C.fidelity.fid.mode = "clean"
_C.fidelity.fid.batch_size = 64

_C.fidelity.fvd = CN()
_C.fidelity.fvd.enabled = False
_C.fidelity.fvd.real_features_path = ""
_C.fidelity.fvd.gen_features_path = ""

_C.fidelity.ssim = False
_C.fidelity.mse_psnr = False

_C.fidelity.lpips = CN()
_C.fidelity.lpips.enabled = False
_C.fidelity.lpips.net = "alex"

_C.fidelity.semantic_alignment = CN()
_C.fidelity.semantic_alignment.enabled = False
_C.fidelity.semantic_alignment.model_name = "openai/clip-vit-base-patch32"
_C.fidelity.semantic_alignment.threshold = 0.25
_C.fidelity.semantic_alignment.batch_size = 32

# ══════════════════════════════════════════════════════════
# Diversity sub-config
# ══════════════════════════════════════════════════════════
_C.diversity = CN()
_C.diversity.agent_diversity = True
_C.diversity.road_diversity = True

_C.diversity.vbench = CN()
_C.diversity.vbench.enabled = False
_C.diversity.vbench.full_json_dir = ""
_C.diversity.vbench.output_path = "./vbench_results"
_C.diversity.vbench.dimensions = []

# ══════════════════════════════════════════════════════════
# Safety sub-config
# ══════════════════════════════════════════════════════════
_C.safety = CN()
_C.safety.rvr = True
_C.safety.rvr_by_type = True
_C.safety.route_incompleteness = True
_C.safety.mscr = True
_C.safety.kinematics = True
_C.safety.safety_satisfaction = True
_C.safety.hazard_rejection = True
_C.safety.ttc = True
_C.safety.mdc = True

_C.kinematics = CN()
_C.kinematics.dt = 0.1
_C.kinematics.hard_brake_threshold = 4.0


def get_cfg(config_file_path):
    config = _C.clone()
    if config_file_path:
        config.merge_from_file(config_file_path)
    if config.output_dir and not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)
    return config
