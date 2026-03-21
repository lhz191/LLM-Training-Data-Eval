# Text-based GUI Agent Evaluation

Task-specific (Layer 2) evaluation for text-based web navigation agent datasets.

## Supported Datasets

| Dataset | Description | Source | Size |
|---------|------------|--------|------|
| **Mind2Web** | Real website tasks with HTML snapshots and human action annotations | [paper](https://arxiv.org/abs/2306.06070) / [data](https://huggingface.co/datasets/osunlp/Mind2Web) | 2,350 tasks |
| **WebShop** | Simulated e-commerce with text state + available actions | [paper](https://arxiv.org/abs/2207.01206) / [data](https://github.com/princeton-nlp/WebShop) | 1,571 trajectories |
| **WebLINX** | Real website multi-turn conversations with HTML | [paper](https://arxiv.org/abs/2402.05930) / [data](https://huggingface.co/datasets/McGill-NLP/WebLINX) | 100k+ interactions |

## Layer 2 Metrics

| Metric | File | Needs Executor | Description |
|--------|------|---------------|-------------|
| Format Check | [metrics/format_check.py](metrics/format_check.py) | Yes | Action format, operation type, target element validity |
| Static Executability | [metrics/static_executability.py](metrics/static_executability.py) | Yes | Action grounding against HTML state |
| Dynamic Executability | [metrics/dynamic_executability.py](metrics/dynamic_executability.py) | Yes | Live website replay (Playwright) |
| HTML Retention | [metrics/html_retention.py](metrics/html_retention.py) | Yes | Target element locatability in raw vs cleaned HTML |
| Trajectory Validity | [metrics/trajectory_validity.py](metrics/trajectory_validity.py) | No | LLM judge: action sequence coherence and completeness |
| Task Complexity | [metrics/task_complexity.py](metrics/task_complexity.py) | No | Action chain length, branching, recovery patterns |
| Diversity | [metrics/diversity.py](metrics/diversity.py) | No | Page structure, action pattern, trajectory, domain diversity |

## Quick Start

```bash
python evaluate.py gui mind2web all
python evaluate.py gui webshop format_check
./submit_eval.sh gui weblinx all
```
