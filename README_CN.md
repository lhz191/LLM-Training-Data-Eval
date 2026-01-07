<div align="center">

# 🔬 LLM 训练数据评估框架

**一个全面的 LLM 训练数据质量评估框架**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README.md) | [中文](README_CN.md)

</div>

---

## 📖 概述

本框架为 LLM 训练数据提供系统化的评估指标，涵盖两大领域：
- **数学推理** - 评估数学问题求解数据的质量
- **API Agent** - 评估工具调用和 API 交互数据的质量

## 📁 项目结构

```
├── Symbolic_and_Logical_Data/     # 数学推理数据
│   └── math_eval/
│       ├── data_types.py          # 数据类型定义
│       ├── loaders.py             # 数据加载器
│       ├── code_executor.py       # 代码执行基类
│       ├── lila_executor.py       # LILA 数据集执行器
│       ├── openmath_executor.py   # OpenMath 数据集执行器
│       ├── format_check.py        # 格式检查
│       ├── validity.py            # 代码有效性检查
│       ├── reasoning_validity.py  # 推理有效性检查
│       ├── faithfulness.py        # 答案忠实性检查
│       ├── diversity.py           # 多样性指标
│       ├── run_full_test.py       # 统一入口
│       ├── embeddings/            # 缓存的 embeddings
│       ├── models/                # 预下载的模型
│       └── results/               # 评估结果输出
│
└── Agent_Data/                    # API Agent 数据
    └── api_agent_eval/
        ├── data_types.py          # 数据类型定义
        ├── loaders.py             # 数据加载器 (ToolBench, xLAM)
        ├── api_executor.py        # 执行器基类
        ├── toolbench_executor.py  # ToolBench 执行器
        ├── xlam_executor.py       # xLAM 执行器
        ├── format_check.py        # 格式检查
        ├── executability.py       # 静态可执行性检查
        ├── dynamic_executability.py  # 动态 API 调用检查
        ├── diversity.py           # 多样性指标
        ├── run_full_test.py       # 统一入口
        ├── embeddings/            # 缓存的 embeddings
        ├── models/                # 预下载的模型
        └── results/               # 评估结果输出
```

## 📊 支持的数据集

| 领域 | 数据集 | 描述 |
|------|--------|------|
| 数学 | LILA | 多样化数学推理数据集 |
| 数学 | OpenMathInstruct-1 | 大规模数学指令数据集 |
| 数学 | NuminaMath-CoT | 链式思维数学推理数据集 |
| Agent | ToolBench | 工具调用数据集 |
| Agent | xLAM-60k | API 交互数据集 |

## 🎯 评估指标

### 数学推理

| 指标 | 描述 |
|------|------|
| **Format Check** | 验证结构正确性 |
| **Validity** | 代码可执行性和正确性 |
| **Reasoning Validity** | 推理过程逻辑验证 |
| **Faithfulness** | 答案与推理的一致性 |
| **Diversity** | 数据多样性 (Vendi Score / KNN) |

### API Agent

| 指标 | 描述 |
|------|------|
| **Format Check** | 验证结构正确性 |
| **Executability** | 静态 API 调用验证 |
| **Dynamic Executability** | 真实 API 调用测试 |
| **Diversity** | 数据多样性 (Vendi Score / KNN) |

## 🚀 快速开始

### 数学推理评估

```bash
cd Symbolic_and_Logical_Data/math_eval

# 格式检查
python run_full_test.py -d lila -m format_check

# 代码有效性
python run_full_test.py -d lila -m validity

# 推理有效性
python run_full_test.py -d lila -m reasoning_validity

# 多样性 (KNN)
python run_full_test.py -d lila -m diversity --diversity-method knn

# 多样性 (Vendi Score)
python run_full_test.py -d lila -m diversity --diversity-method vendi
```

### API Agent 评估

```bash
cd Agent_Data/api_agent_eval

# 格式检查（并行）
python run_full_test.py -d toolbench -m format_check --parallel

# 静态可执行性（并行）
python run_full_test.py -d toolbench -m executability --parallel

# 动态可执行性（需要 RapidAPI Key）
export RAPIDAPI_KEY="your_key"
python run_full_test.py -d toolbench -m dynamic_executability

# 多样性 (KNN)
python run_full_test.py -d toolbench -m diversity --diversity-method knn

# 多样性 (Vendi Score)
python run_full_test.py -d toolbench -m diversity --diversity-method vendi
```

## 📝 Slurm 提交

```bash
# 数学推理
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

## 📦 依赖

```
torch>=2.0
transformers>=4.30
sentence-transformers>=2.2
numpy>=1.24
tqdm>=4.65
```

## 📄 许可证

MIT License

