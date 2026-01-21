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

### 1. 数据类型：固定 + 可扩展设计

`data_types.py` 是专门为每种数据类型特别设计的数据结构，采用**固定 + 可扩展**的设计模式。

```python
# 示例：API Agent 数据类型 (api_agent_eval/data_types.py)
@dataclass
class APIAgentSample:
    # === 固定字段 ===
    query: str              # 用户查询/指令
    tools: List[Tool]       # 可用工具/API
    conversations: List[Message]  # 交互历史
    
    # === 可扩展字段 ===
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**设计理念：**

- **固定字段**：这些字段是为每类评测的数据类型针对性确定的，能够反映该类数据的**真正特点**，代表了这类数据的**核心价值**。不同数据集在这些字段上是共通的。

- **可扩展字段 (`metadata`)**：考虑到不同数据集都有自己的特殊性，每个数据集在实际使用时可能有一些其他数据集没有的特殊字段。这些字段可以存储在 `metadata` 中，供特定数据集使用，保证了框架的可扩展性。

**使用示例：**
```python
# ToolBench 有额外字段如 'answer_generation'，这是其他数据集没有的
sample = APIAgentSample(
    query="搜索天气",
    tools=[...],
    conversations=[...],
    metadata={
        "answer_generation": {...},  # ToolBench 特有字段
        "category": "weather",       # ToolBench 特有字段
    }
)
```

### 2. 执行器：基类 + 数据集实现

每个领域有一个**基类执行器**定义接口，以及**数据集特定实现**。

**为什么需要数据集特定执行器？**
- 不同数据集有不同的验证需求
- 需要检查的内容因数据集而异（如 ToolBench 需要验证 API 参数，xLAM 需要验证函数模式）
- 某些数据集有需要自定义检查逻辑的特有字段

**何时使用预置执行器：**
- 如果你的数据集与现有数据集相似（如与 ToolBench 格式相同）
- 如果你的验证需求足够通用

**何时编写自定义执行器：**
- 你的数据集有独特的验证需求
- 你需要检查存储在 `metadata` 中的数据集特定字段

```python
# 基类 (api_agent_eval/api_executor.py)
class FormatChecker(ABC):
    @abstractmethod
    def check(self, sample: APIAgentSample) -> Tuple[bool, List[str]]:
        """检查样本格式是否有效。返回 (is_valid, error_messages)。"""
        pass
```

```python
# 数据集特定实现 (api_agent_eval/toolbench_executor.py)
class ToolBenchFormatChecker(FormatChecker):
    def check(self, sample: APIAgentSample) -> Tuple[bool, List[str]]:
        errors = []
        # ToolBench 特定验证逻辑
        if not sample.query:
            errors.append("缺少 query")
        if not sample.tools:
            errors.append("缺少 tools")
        # ... 更多检查
        return len(errors) == 0, errors
```

### 3. 添加新数据集支持

要评估新数据集，你可以：
- **使用现有加载器/执行器**（如果你的数据集格式与已支持的相似）
- **创建自定义加载器/执行器**（如果你的数据集有独特需求）

#### 步骤 1：创建加载器（在 `loaders.py` 中）- *可选*

加载器的主要目的是**将数据集字段与评估数据类型对齐**。

**如果数据集使用标准字段名**（如 `query`、`tools`、`video`、`text`、`image_path`、`caption`），可以直接使用预置的 `GeneralLoader`：

```python
# 无需自定义加载器 - 直接使用 GeneralLoader
from loaders import GeneralLoader
loader = GeneralLoader('/path/to/your_dataset.jsonl')
```

**如果数据集有不同字段名或需要自定义解析**，编写自定义加载器：

```python
# 添加到 loaders.py
from data_types import APIAgentSample

class MyDatasetLoader(BaseLoader):
    """
    MyDataset 加载器。
    将数据集特定字段映射到 APIAgentSample。
    """
    def iterate(self) -> Iterator[APIAgentSample]:
        with open(self.data_path) as f:
            for line in f:
                data = json.loads(line)
                # 将你的数据集字段映射到标准字段
                yield APIAgentSample(
                    query=data['instruction'],      # 你的字段 -> 标准字段
                    tools=self._parse_tools(data['functions']),
                    conversations=self._parse_conversations(data['messages']),
                    metadata={
                        'custom_field': data.get('custom_field'),  # 数据集特定
                    }
                )
```

#### 步骤 2：创建数据集特定执行器

```python
# my_dataset_executor.py
from api_executor import FormatChecker, ExecutabilityChecker

class MyDatasetFormatChecker(FormatChecker):
    def check(self, sample: APIAgentSample) -> Tuple[bool, List[str]]:
        errors = []
        # 你的数据集特定验证逻辑
        if not sample.query:
            errors.append("缺少 query")
        # ... 针对你数据集的自定义检查
        return len(errors) == 0, errors
```

#### 步骤 3：在 run_full_test.py 中注册

```python
# 添加到 DATASETS 配置
DATASETS = {
    'my_dataset': {
        'name': 'My Dataset',
        'data_path': '/path/to/my_dataset.jsonl',
        'loader_class': MyDatasetLoader,
        'format_checker': MyDatasetFormatChecker,
        # ... 其他配置
    },
}
```

#### 步骤 4：运行评估

```bash
python run_full_test.py -d my_dataset -m format_check
python run_full_test.py -d my_dataset -m all
```

### 4. 指标计算：固定逻辑

`metrics/` 中的指标计算逻辑是**固定且可复用的**。一旦你提供了正确的数据加载器和执行器，指标会自动计算：

```python
# metrics/format_check.py - 适用于任何数据集
def compute_format_check(
    data_iterator: Iterator[Sample],
    format_checker: FormatChecker,  # 你的数据集特定检查器
    ...
) -> Dict[str, Any]:
    # 固定的计算逻辑
    for sample in data_iterator:
        is_valid, errors = format_checker.check(sample)
        # ... 累积结果
    return results
```

**关键优势：** 你只需要实现数据集特定的加载器和执行器。指标计算、结果保存和报告由框架自动处理。

### 总结

| 组件 | 作用 | 用户操作 |
|------|------|----------|
| `data_types.py` | 定义数据结构 | 直接使用（通过 `metadata` 扩展） |
| `loaders.py` | 加载数据到标准格式 | 为新数据集实现 |
| `*_executor.py` | 检查器基类 | 继承并实现 |
| `metrics/` | 计算指标 | 直接使用（无需修改） |
| `scripts/` | 入口脚本 | 添加数据集配置 |

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
