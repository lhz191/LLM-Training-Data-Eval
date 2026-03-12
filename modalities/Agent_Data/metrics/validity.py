"""
Validity — Action Executability, Success Rate, Percent Complete

Survey Section 7.2 (Validity):

    ExecRate = (1 / Σ Tᵢ) Σᵢ Σₜ 𝟙{aₜ executes without error in sₜ}

    SRvalid  = (1/N) Σᵢ 𝟙{all task goals satisfied in episode i}

    PC(τᵢ)  = |Ĝ(τᵢ)| / |G|

References:
    - PARTNR (Chang et al., 2024)
    - CARLA text-to-traffic (Ruan et al., 2025)
"""

from typing import List, Dict, Any, Callable, Optional


def compute_exec_rate(
    data: List[Dict[str, Any]],
    exec_checker: Optional[Callable[[Dict, Dict], bool]] = None,
) -> Dict[str, Any]:
    """
    ExecRate = (1 / Σ Tᵢ) Σᵢ Σₜ 𝟙{aₜ executes without error in sₜ}

    Args:
        data: list of trajectories, each with "actions" and "states"
        exec_checker: exec_checker(action, state) -> bool
                      If None, reads action["executed"] from pre-computed data.
    """
    total_actions = 0
    successful_actions = 0

    for traj in data:
        actions = traj.get("actions", [])
        states = traj.get("states", [None] * len(actions))
        for act, st in zip(actions, states):
            total_actions += 1
            if exec_checker is not None:
                if exec_checker(act, st):
                    successful_actions += 1
            else:
                if act.get("executed", False):
                    successful_actions += 1

    rate = successful_actions / total_actions if total_actions > 0 else 0.0
    return {
        "exec_rate": rate,
        "successful_actions": successful_actions,
        "total_actions": total_actions,
    }


def compute_success_rate(
    data: List[Dict[str, Any]],
    goal_checker: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Dict[str, Any]:
    """
    SRvalid = (1/N) Σᵢ 𝟙{all task goals satisfied in episode i}

    Args:
        data: list of trajectories
        goal_checker: goal_checker(trajectory) -> bool
                      If None, reads "success" field or derives from
                      "goals" vs "goals_achieved".
    """
    n = 0
    successes = 0

    for traj in data:
        if goal_checker is not None:
            n += 1
            if goal_checker(traj):
                successes += 1
        elif "success" in traj:
            n += 1
            if traj["success"]:
                successes += 1
        elif traj.get("goals"):
            n += 1
            goals = set(traj["goals"])
            achieved = set(traj.get("goals_achieved", []))
            if goals.issubset(achieved):
                successes += 1

    rate = successes / n if n > 0 else 0.0
    return {
        "success_rate": rate,
        "successes": successes,
        "total_episodes": n,
    }


def compute_percent_complete(
    data: List[Dict[str, Any]],
    goal_checker: Optional[Callable[[Dict[str, Any]], float]] = None,
) -> Dict[str, Any]:
    """
    PC(τᵢ) = |Ĝ(τᵢ)| / |G|

    Args:
        data: list of trajectories
        goal_checker: goal_checker(trajectory) -> float (fraction in [0, 1])
                      If None, reads "goals" / "goals_achieved" or
                      pre-computed "percent_complete".
    """
    scores = []

    for traj in data:
        if goal_checker is not None:
            scores.append(goal_checker(traj))
        elif "percent_complete" in traj:
            scores.append(float(traj["percent_complete"]))
        else:
            goals = traj.get("goals", [])
            achieved = traj.get("goals_achieved", [])
            if not goals:
                continue
            achieved_valid = set(achieved) & set(goals)
            scores.append(len(achieved_valid) / len(goals))

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "percent_complete_mean": avg,
        "num_episodes": len(scores),
    }
