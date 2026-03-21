# Image-to-Report Evaluation

Task-specific (Layer 2) evaluation for image-conditioned report generation datasets (medical reports, detailed image descriptions).

## Supported Datasets

| Dataset | Description | Source | Size |
|---------|------------|--------|------|
| **IU X-Ray** | Chest X-ray radiology report generation | [paper](https://openi.nlm.nih.gov/) / [data](https://huggingface.co/datasets/IUXRAY) | ~3.9k reports |
| **ShareGPT4V** | Detailed image descriptions (report subset) | [paper](https://arxiv.org/abs/2311.12793) / [data](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) | ~100k |

## Layer 2 Metrics

| Metric | File | Needs Executor | Description |
|--------|------|---------------|-------------|
| Format Check | [metrics/format_check.py](metrics/format_check.py) | Yes | Image path existence, instruction/report field validity |
| Validity | [metrics/validity.py](metrics/validity.py) | Yes | Image loadability, report non-emptiness, structural completeness |
| Report Quality | [metrics/report_quality.py](metrics/report_quality.py) | Yes | Clinical accuracy, hallucination detection, completeness (LLM judge) |
| Duplication | [metrics/duplication.py](metrics/duplication.py) | No | Near-duplicate report detection |
| Diversity | [metrics/diversity.py](metrics/diversity.py) | No | Report topic, vocabulary, instruction pattern diversity |

## Quick Start

```bash
python evaluate.py report iu_xray format_check
python evaluate.py report sharegpt4v all
./submit_eval.sh report iu_xray all
```
