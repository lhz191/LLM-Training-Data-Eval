# API Agent Evaluation

Task-specific (Layer 2) evaluation for API function-calling agent datasets.

## Supported Datasets

| Dataset | Description | Source | Size |
|---------|------------|--------|------|
| **ToolBench** | Multi-turn real API calling with RapidAPI | [paper](https://arxiv.org/abs/2307.16789) / [data](https://github.com/OpenBMB/ToolBench) | ~12k |
| **xLAM-60k** | Single-turn synthetic function calling | [paper](https://arxiv.org/abs/2411.10440) / [data](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 60k |
| **Arcee Agent Data** | Mixed-format agent trajectories (5 sub-datasets) | [data](https://huggingface.co/datasets/arcee-ai/agent-data) | ~50k |

## Layer 2 Metrics

| Metric | File | Needs Executor | Description |
|--------|------|---------------|-------------|
| Format Check | [metrics/format_check.py](metrics/format_check.py) | Yes | Structural validity per dataset format |
| Executability | [metrics/executability.py](metrics/executability.py) | Yes | API name existence, required params, type matching, LLM judge |
| Dynamic Executability | [metrics/dynamic_executability.py](metrics/dynamic_executability.py) | Yes | Real API call testing via RapidAPI |
| Diversity | [metrics/diversity.py](metrics/diversity.py) | No | 7-dimension diversity (embedding, Self-BLEU, call patterns, params, etc.) |
| Task Complexity | [metrics/task_complexity.py](metrics/task_complexity.py) | No | Tool selection / param filling / multi-step planning difficulty |
| Trustworthy | [metrics/trustworthy.py](metrics/trustworthy.py) | No | Safety via AgentDoG guard model |

## Quick Start

```bash
# Via unified entry
python evaluate.py api toolbench format_check
python evaluate.py api xlam executability
python evaluate.py api toolbench all

# Via job submission
./submit_eval.sh api toolbench all
```
