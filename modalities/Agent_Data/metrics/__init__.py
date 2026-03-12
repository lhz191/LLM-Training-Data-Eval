"""
Agent Data Metrics — Survey Sections 7.2 & 7.3

Quality:
  Validity  — ExecRate, SRvalid, PC
  Fidelity  — FID, FVD, SSIM, MSE/PSNR, LPIPS, SemAlign/SemFid/MatchRate
  Diversity — AD, RD, VBench

Trustworthy:
  Safety    — RVR, RVR(t), RI, MSCR, Kinematics (ACC/YV/JerkRMS/HardBrake),
              SafetySat, Rejection/Risk, TTC, MDC
"""

from .validity import (
    compute_exec_rate,
    compute_success_rate,
    compute_percent_complete,
)
from .fidelity import (
    compute_fid,
    compute_frechet_distance,
    compute_fvd,
    compute_ssim,
    compute_mse_psnr,
    compute_lpips,
    compute_semantic_alignment,
)
from .diversity import (
    compute_agent_diversity,
    compute_road_diversity,
    compute_vbench,
)
from .safety import (
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

__all__ = [
    # Validity
    "compute_exec_rate",
    "compute_success_rate",
    "compute_percent_complete",
    # Fidelity
    "compute_fid",
    "compute_frechet_distance",
    "compute_fvd",
    "compute_ssim",
    "compute_mse_psnr",
    "compute_lpips",
    "compute_semantic_alignment",
    # Diversity
    "compute_agent_diversity",
    "compute_road_diversity",
    "compute_vbench",
    # Safety
    "compute_rvr",
    "compute_rvr_by_type",
    "compute_route_incompleteness",
    "compute_mscr",
    "compute_kinematics",
    "compute_safety_satisfaction",
    "compute_hazard_rejection",
    "compute_ttc",
    "compute_mdc",
]
