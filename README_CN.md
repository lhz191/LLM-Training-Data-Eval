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

本框架为 LLM 训练数据提供系统化的评估指标，涵盖多种数据模态：

- **符号与逻辑数据** - 数学推理、代码、形式逻辑
- **Agent 数据** - 工具调用、API 交互、网页导航
- **视觉-语言数据** - 图文、视频-文本多模态数据
- **文本数据** - 纯文本语料 *(即将推出)*
- **表格数据** - 结构化表格数据 *(即将推出)*
- **半结构化与图数据** - 知识图谱、半结构化数据 *(即将推出)*

## 📁 项目结构

```
├── Symbolic_and_Logical_Data/         # 符号与逻辑数据
│   └── math_eval/
│       ├── data_types.py              # 数据类型定义
│       ├── loaders.py                 # 数据加载器
│       ├── code_executor.py           # 代码执行基类
│       ├── {dataset}_executor.py      # 数据集执行器
│       ├── metrics/                   # 评估指标
│       ├── scripts/
│       │   └── run_full_test.py       # 统一入口
│       └── results/                   # 评估结果输出
│
├── Agent_Data/                        # Agent 数据
│   └── api_agent_eval/
│       ├── data_types.py              # 数据类型定义
│       ├── loaders.py                 # 数据加载器
│       ├── api_executor.py            # 执行器基类
│       ├── {dataset}_executor.py      # 数据集执行器
│       ├── metrics/                   # 评估指标
│       ├── scripts/
│       │   └── run_full_test.py       # 统一入口
│       └── results/                   # 评估结果输出
│
├── Vision_Language_Data/              # 视觉-语言数据
│   ├── video_text_eval/               # 视频-文本评估
│   │   ├── data_types.py              # VideoTextSample 定义
│   │   ├── loaders.py                 # 数据加载器
│   │   ├── metrics/                   # 评估指标
│   │   ├── scripts/
│   │   │   └── run_full_test.py       # 统一入口
│   │   └── results/                   # 评估结果输出
│   │
│   └── image_text_eval/               # 图像-文本评估
│       ├── data_types.py              # ImageTextSample 定义
│       ├── loaders.py                 # 数据加载器
│       ├── image_executor.py          # 格式检查器基类
│       ├── {dataset}_executor.py      # 数据集执行器
│       ├── metrics/                   # 评估指标
│       ├── scripts/
│       │   └── run_full_test.py       # 统一入口
│       └── results/                   # 评估结果输出
│
├── Text_Data/                         # 文本数据（即将推出）
│
├── Tabular_Data/                      # 表格数据（即将推出）
│
└── Semi_Structured_Graph_Data/        # 半结构化与图数据（即将推出）
```

## 📊 支持的数据集

| 领域 | 数据集 | 描述 |
|------|--------|------|
| 数学 | LILA | 多样化数学推理数据集 |
| 数学 | OpenMathInstruct-1 | 大规模数学指令数据集 |
| Agent | ToolBench | 工具调用数据集 |
| Agent | xLAM-60k | API 交互数据集 |
| 视频-文本 | 通用 JSONL | 视频-文本对 |
| 图像-文本 | COCO Caption | 图像描述数据集 |

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

### 视频-文本

| 指标 | 描述 |
|------|------|
| **Frame Diversity** | 基于光流的帧多样性 |
| **Semantic Diversity** | Inception V3 特征多样性 |
| **Object Consistency** | 基于 CLIP 的对象一致性 |
| **Cross-Modal Consistency** | ViCLIP 视频-文本对齐 |
| **Safety Bench** | GPT-4 Vision 安全性评估 |
| **Holistic Fidelity** | VBench 综合评估 |

### 图像-文本

| 指标 | 描述 |
|------|------|
| **Inception Score** | 图像质量与多样性 |
| **Prompt Fidelity** | CLIP 图文对齐度 |
| **Well-Formed Rate** | 数据格式正确率 |
| **C2PA Validation** | 内容来源验证 |

## 🚀 快速开始

### 数学推理评估

```bash
cd Symbolic_and_Logical_Data/math_eval/scripts

# 格式检查
python run_full_test.py -d lila -m format_check

# 代码有效性
python run_full_test.py -d lila -m validity

# 多样性 (KNN)
python run_full_test.py -d lila -m diversity --diversity-method knn
```

### API Agent 评估

```bash
cd Agent_Data/api_agent_eval/scripts

# 格式检查（并行）
python run_full_test.py -d toolbench -m format_check --parallel

# 静态可执行性（并行）
python run_full_test.py -d toolbench -m executability --parallel

# 动态可执行性（需要 RapidAPI Key）
export RAPIDAPI_KEY="your_key"
python run_full_test.py -d toolbench -m dynamic_executability
```

### 视频-文本评估

```bash
cd Vision_Language_Data/video_text_eval/scripts

# 帧多样性
python run_full_test.py -d test -m frame_diversity

# 语义多样性
python run_full_test.py -d test -m semantic_diversity

# 跨模态一致性
python run_full_test.py -d test -m cross_modal_consistency

# 运行所有指标
python run_full_test.py -d test -m all
```

### 图像-文本评估

```bash
cd Vision_Language_Data/image_text_eval/scripts

# Inception Score
python run_full_test.py -d coco_caption -m inception_score

# 图文对齐度
python run_full_test.py -d coco_caption -m prompt_fidelity

# 格式正确率
python run_full_test.py -d coco_caption -m well_formed_rate

# 运行所有指标
python run_full_test.py -d coco_caption -m all
```

## 🛠️ 架构与可扩展性

本框架设计时充分考虑了**可扩展性**。每个评估领域都遵循一致的架构模式，便于用户轻松添加对新数据集的支持。

### 核心组件

```
{domain}_eval/
├── data_types.py      # 固定 + 可扩展的数据结构
├── loaders.py         # 基类加载器 + 数据集特定加载器
├── {domain}_executor.py   # 检查器基类
├── {dataset}_executor.py  # 数据集特定实现
├── metrics/           # 指标计算（固定逻辑）
└── scripts/           # 入口脚本
```

### 数据类型：固定 + 可扩展设计

每个领域都有精心设计的数据类型，捕捉该数据类别的**核心特征**：

```python
@dataclass
class APIAgentSample:
    # === 固定字段（代表数据的核心价值）===
    query: str              # 用户查询/指令
    tools: List[Tool]       # 可用工具/API
    conversations: List[Message]  # 交互历史
    
    # === 可扩展字段（用于数据集特定需求）===
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**设计理念：**
- **固定字段**代表定义此数据类型价值的基本特征
- **`metadata` 字段**允许存储其他数据集可能没有的特定字段

### 添加新数据集支持

1. **创建加载器**（在 `loaders.py` 中）- 将数据集字段映射到标准数据类型
2. **创建执行器**（在 `{dataset}_executor.py` 中）- 实现数据集特定的验证逻辑
3. **注册到配置** - 在 `run_full_test.py` 的 `DATASETS` 中添加配置
4. **运行评估** - 使用统一命令运行

详细指南请参阅 [README.md](README.md) 的架构部分。

## 📦 依赖

```
torch>=2.0
transformers>=4.30
sentence-transformers>=2.2
numpy>=1.24
tqdm>=4.65
clip
opencv-python
torchvision
```

## 📄 许可证

MIT License
