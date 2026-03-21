# Agent Data — Layer 1 Metrics

Modality-level metrics that apply to **all** agent data (API agents, GUI agents, etc.), regardless of downstream task. These audit individual data points before they form task-specific datasets.

## Metrics

| Metric | File | What it measures | Reference |
|--------|------|-----------------|-----------|
| **Agent Diversity (AD)** | [diversity.py](diversity.py) | Ratio of unique agent behaviors (type, action, position) to total agents | Ruan et al., 2025 |
| **Road Diversity (RD)** | [diversity.py](diversity.py) | Ratio of unique road identifiers to total roads | Ruan et al., 2025 |
| **VBench Diversity** | [diversity.py](diversity.py) | Video/world-model rollout diversity | Huang et al., 2024b |
| **FID** | [fidelity.py](fidelity.py) | Frechet Inception Distance between real and generated image sets | Heusel et al., 2018 |
| **FVD** | [fidelity.py](fidelity.py) | Frechet Video Distance on I3D features | Unterthiner et al., 2019 |
| **SSIM** | [fidelity.py](fidelity.py) | Structural Similarity Index between paired images | Wang et al., 2004 |
| **MSE / PSNR** | [fidelity.py](fidelity.py) | Mean Squared Error and Peak Signal-to-Noise Ratio | — |
| **LPIPS** | [fidelity.py](fidelity.py) | Learned Perceptual Image Patch Similarity | Zhang et al., 2018 |
| **SemAlign / SemFid** | [fidelity.py](fidelity.py) | CLIP-based semantic alignment between prompt and outcome | Radford et al., 2021 |
| **Rule Violation Rate (RVR)** | [safety.py](safety.py) | Fraction of violated traffic/safety rules per episode | SafeBench (Xu et al., 2022; Zhang et al., 2024b) |
| **Route Incompleteness (RI)** | [safety.py](safety.py) | 1 - distance completed / planned route length | SafeBench |
| **Speed Compliance (MSCR)** | [safety.py](safety.py) | Fraction of timesteps meeting minimum speed | SafeBench |
| **Kinematics (ACC, Yaw, Jerk)** | [safety.py](safety.py) | Acceleration, yaw velocity, jerk smoothness | Ward et al., 2015 |
| **Safety Satisfaction** | [safety.py](safety.py) | Fraction of trajectories satisfying formal safety spec (LTL/STL) | SELP (Wu et al., 2025); T3 Planner (Li & Zhao, 2025) |
| **Hazard Rejection / Risk** | [safety.py](safety.py) | Agent's refusal rate on hazardous instructions | SafeAgentBench (Yin et al., 2025) |
| **TTC / MDC** | [safety.py](safety.py) | Time-to-Collision and Minimum Distance to Collision | Ward et al., 2015; Gao et al., 2025 |
| **ExecRate** | [validity.py](validity.py) | Fraction of actions that execute without error | PARTNR (Chang et al., 2024) |
| **Success Rate** | [validity.py](validity.py) | Fraction of episodes where all task goals are satisfied | PARTNR (Chang et al., 2024) |
| **Percent Complete (PC)** | [validity.py](validity.py) | Fraction of task goals completed per episode | Ruan et al., 2025 |

## Relationship to Layer 2

These Layer 1 metrics provide **modality-universal** quality signals. Each downstream task under Agent Data (e.g., `api_agent_eval`, `text_gui_agent_eval`) has its own Layer 2 metrics in its `metrics/` directory that provide **task-specific** checks.

See:
- [api_agent_eval/metrics/](../api_agent_eval/metrics/) — format_check, executability, diversity, task_complexity, trustworthy
- [text_gui_agent_eval/metrics/](../text_gui_agent_eval/metrics/) — format_check, static_executability, html_retention, trajectory_validity, task_complexity, diversity
