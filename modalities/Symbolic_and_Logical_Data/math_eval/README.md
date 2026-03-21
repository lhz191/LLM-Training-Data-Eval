# Math Reasoning Evaluation

Task-specific (Layer 2) evaluation for mathematical reasoning datasets (CoT and code-based solutions).

## Supported Datasets

| Dataset | Description | Source | Size |
|---------|------------|--------|------|
| **OpenMathInstruct-1** | Large-scale math instructions with code solutions | [paper](https://arxiv.org/abs/2402.10176) / [data](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) | 1.8M |
| **LILA** | Diverse math reasoning with multiple Python programs per problem | [paper](https://arxiv.org/abs/2210.17517) / [data](https://github.com/allenai/LILA) | ~130k |

## Layer 2 Metrics

| Metric | File | Needs Executor | Description |
|--------|------|---------------|-------------|
| Format Check | [metrics/format_check.py](metrics/format_check.py) | Yes | Solution structure, code extractability, answer presence |
| Validity | [metrics/validity.py](metrics/validity.py) | Yes | Code execution in sandbox + answer comparison with ground truth |
| Faithfulness | [metrics/faithfulness.py](metrics/faithfulness.py) | No | Solution-answer consistency (does the solution actually derive the claimed answer) |
| Reasoning Validity | [metrics/reasoning_validity.py](metrics/reasoning_validity.py) | Yes | LLM judge: reasoning process correctness + result matching |
| Diversity | [metrics/diversity.py](metrics/diversity.py) | No | Question topic, solution strategy, answer distribution diversity |

## Quick Start

```bash
python evaluate.py math lila format_check
python evaluate.py math openmathinstruct validity
./submit_eval.sh math lila all
```
