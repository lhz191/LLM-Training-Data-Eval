"""
Diversity — Agent Diversity (AD), Road Diversity (RD), VBench

Survey Section 7.2 (Diversity):

  AD = |unique agents| / |total agents|
  RD = |unique roads|  / |total roads|

  where agent uniqueness is defined by (type, action, relative position)
  tuples, and road uniqueness is based on road identifiers (Ruan et al., 2025).

  For video / world-model rollout diversity (VBench, Huang et al., 2024b),
  reuses the full VBench integration from:
      LLMDataBenchmark_wjd/Multimodal/metrics/vbench/
"""

import os
import sys
from typing import List, Dict, Any, Optional


# ═══════════════════════════════════════════════════════════════
# AD — Agent Diversity (Ruan et al., 2025)
# ═══════════════════════════════════════════════════════════════

def compute_agent_diversity(
    trajectories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Agent Diversity (AD) = |unique agents| / |total agents|

    Uniqueness defined by (type, action, position) tuples.
    Trajectories should be repeated generations for the same prompt/scenario
    to measure diversity across generations (Ruan et al., 2025).

    Expected fields per trajectory:
        "agents": list of dicts with "type", "action", "position"
    """
    all_agents = []
    for traj in trajectories:
        for agent in traj.get("agents", []):
            key = (
                agent.get("type", ""),
                agent.get("action", ""),
                str(agent.get("position", "")),
            )
            all_agents.append(key)

    if not all_agents:
        return {"agent_diversity": 0.0, "unique": 0, "total": 0}

    unique = len(set(all_agents))
    total = len(all_agents)
    return {
        "agent_diversity": unique / total,
        "unique": unique,
        "total": total,
    }


# ═══════════════════════════════════════════════════════════════
# RD — Road Diversity (Ruan et al., 2025)
# ═══════════════════════════════════════════════════════════════

def compute_road_diversity(
    trajectories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Road Diversity (RD) = |unique roads| / |total roads|

    Uniqueness based on road identifiers across repeated generations
    for the same prompt/scenario (Ruan et al., 2025).

    Expected fields per trajectory:
        "roads": list of road identifiers (str or int)
    """
    all_roads = []
    for traj in trajectories:
        all_roads.extend(traj.get("roads", []))

    if not all_roads:
        return {"road_diversity": 0.0, "unique": 0, "total": 0}

    unique = len(set(all_roads))
    total = len(all_roads)
    return {
        "road_diversity": unique / total,
        "unique": unique,
        "total": total,
    }


# ═══════════════════════════════════════════════════════════════
# VBench — Huang et al., 2024b
# Reuses LLMDataBenchmark_wjd/Multimodal/metrics/vbench/
# ═══════════════════════════════════════════════════════════════

def compute_vbench(
    video_paths: List[str],
    full_info_dir: str,
    output_path: str = "./vbench_results",
    dimensions: Optional[List[str]] = None,
    device: str = None,
) -> Dict[str, Any]:
    """
    VBench video generation evaluation via Multimodal's VBench integration.

    Args:
        video_paths: list of paths to generated video files
        full_info_dir: path to VBench_full_info.json
        output_path: directory to save evaluation results
        dimensions: list of dimensions to evaluate (None = all 16)
        device: torch device
    """
    import torch

    _MULTIMODAL_ROOT = os.path.join(
        os.path.dirname(__file__), "..", "..", "Multimodal"
    )
    sys.path.insert(0, _MULTIMODAL_ROOT)
    from metrics.vbench import VBench

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bench = VBench(device, full_info_dir, output_path)

    video_data = [{"video": p} for p in video_paths]

    if dimensions is None:
        dimensions = bench.build_full_dimension_list()

    results = bench.evaluate(
        video_data,
        name="agent_vbench",
        dimension_list=dimensions,
        mode="custom_input",
    )

    return {
        "vbench_scores": results,
        "dimensions_evaluated": dimensions,
        "num_videos": len(video_paths),
    }
