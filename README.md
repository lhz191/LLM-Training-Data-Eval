<div align="center">

# 🔬 LLM Training Data Evaluation

**A comprehensive framework for evaluating LLM training data quality**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README.md) | [中文](README_CN.md)

</div>

---

## 📖 Overview

This framework provides systematic evaluation metrics for LLM training data across two domains:
- **Mathematical Reasoning** - Evaluating math problem-solving data quality
- **API Agent** - Evaluating tool-calling and API interaction data quality

## 📁 Project Structure

```
├── Symbolic_and_Logical_Data/     # Mathematical Reasoning Data
│   └── math_eval/
│       ├── data_types.py          # Data type definitions
│       ├── loaders.py             # Data loaders
│       ├── code_executor.py       # Code execution base class
│       ├── lila_executor.py       # LILA dataset executor
│       ├── openmath_executor.py   # OpenMath dataset executor
│       ├── format_check.py        # Format validation
│       ├── validity.py            # Code validity check
│       ├── reasoning_validity.py  # Reasoning validity check
│       ├── faithfulness.py        # Answer faithfulness check
│       ├── diversity.py           # Diversity metrics
│       ├── run_full_test.py       # Unified entry point
│       ├── embeddings/            # Cached embeddings
│       ├── models/                # Pre-downloaded models
│       └── results/               # Evaluation outputs
│
└── Agent_Data/                    # API Agent Data
    └── api_agent_eval/
        ├── data_types.py          # Data type definitions
        ├── loaders.py             # Data loaders (ToolBench, xLAM)
        ├── api_executor.py        # Executor base class
        ├── toolbench_executor.py  # ToolBench executor
        ├── xlam_executor.py       # xLAM executor
        ├── format_check.py        # Format validation
        ├── executability.py       # Static executability check
        ├── dynamic_executability.py  # Dynamic API call check
        ├── diversity.py           # Diversity metrics
        ├── run_full_test.py       # Unified entry point
        ├── embeddings/            # Cached embeddings
        ├── models/                # Pre-downloaded models
        └── results/               # Evaluation outputs
```

## 📊 Supported Datasets

| Domain | Dataset | Description |
|--------|---------|-------------|
| Math | LILA | Diverse mathematical reasoning |
| Math | OpenMathInstruct-1 | Large-scale math instructions |
| Math | NuminaMath-CoT | Chain-of-thought math reasoning |
| Agent | ToolBench | Tool calling dataset |
| Agent | xLAM-60k | API interaction dataset |

## 🎯 Evaluation Metrics

### Mathematical Reasoning

| Metric | Description |
|--------|-------------|
| **Format Check** | Validates structural correctness |
| **Validity** | Code executability and correctness |
| **Reasoning Validity** | Logical reasoning process validation |
| **Faithfulness** | Answer-reasoning consistency |
| **Diversity** | Data variety (Vendi Score / KNN) |

### API Agent

| Metric | Description |
|--------|-------------|
| **Format Check** | Validates structural correctness |
| **Executability** | Static API call validation |
| **Dynamic Executability** | Real API call testing |
| **Diversity** | Data variety (Vendi Score / KNN) |

## 🚀 Quick Start

### Mathematical Reasoning Evaluation

```bash
cd Symbolic_and_Logical_Data/math_eval

# Format check
python run_full_test.py -d lila -m format_check

# Code validity
python run_full_test.py -d lila -m validity

# Reasoning validity
python run_full_test.py -d lila -m reasoning_validity

# Diversity (KNN)
python run_full_test.py -d lila -m diversity --diversity-method knn

# Diversity (Vendi Score)
python run_full_test.py -d lila -m diversity --diversity-method vendi
```

### API Agent Evaluation

```bash
cd Agent_Data/api_agent_eval

# Format check (parallel)
python run_full_test.py -d toolbench -m format_check --parallel

# Static executability (parallel)
python run_full_test.py -d toolbench -m executability --parallel

# Dynamic executability (requires RapidAPI Key)
export RAPIDAPI_KEY="your_key"
python run_full_test.py -d toolbench -m dynamic_executability

# Diversity (KNN)
python run_full_test.py -d toolbench -m diversity --diversity-method knn

# Diversity (Vendi Score)
python run_full_test.py -d toolbench -m diversity --diversity-method vendi
```

## 📝 Slurm Submission

```bash
# Mathematical Reasoning
cd Symbolic_and_Logical_Data/math_eval
sbatch submit_full_test.sh
sbatch submit_diversity_gpu.sh

# API Agent
cd Agent_Data/api_agent_eval
sbatch submit_format_check.sh
sbatch submit_executability.sh
sbatch submit_diversity_vendi.sh
sbatch submit_diversity_knn.sh
```

## 📦 Requirements

```
torch>=2.0
transformers>=4.30
sentence-transformers>=2.2
numpy>=1.24
tqdm>=4.65
```

## 📄 License

MIT License
