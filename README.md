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

This framework provides systematic evaluation metrics for LLM training data across three domains:
- **Mathematical Reasoning** - Evaluating math problem-solving data quality
- **API Agent** - Evaluating tool-calling and API interaction data quality
- **Vision-Language** - Evaluating multimodal (image-text, video-text) data quality

## 📁 Project Structure

```
├── Symbolic_and_Logical_Data/         # Mathematical Reasoning Data
│   └── math_eval/
│       ├── data_types.py              # Data type definitions
│       ├── loaders.py                 # Data loaders
│       ├── code_executor.py           # Code execution base class
│       ├── lila_executor.py           # LILA dataset executor
│       ├── openmath_executor.py       # OpenMath dataset executor
│       ├── metrics/                   # Evaluation metrics
│       │   ├── format_check.py        # Format validation
│       │   ├── validity.py            # Code validity check
│       │   ├── reasoning_validity.py  # Reasoning validity check
│       │   ├── faithfulness.py        # Answer faithfulness check
│       │   └── diversity.py           # Diversity metrics
│       ├── scripts/
│       │   └── run_full_test.py       # Unified entry point
│       └── results/                   # Evaluation outputs
│
├── Agent_Data/                        # API Agent Data
│   └── api_agent_eval/
│       ├── data_types.py              # Data type definitions
│       ├── loaders.py                 # Data loaders (ToolBench, xLAM)
│       ├── api_executor.py            # Executor base class
│       ├── toolbench_executor.py      # ToolBench executor
│       ├── xlam_executor.py           # xLAM executor
│       ├── metrics/                   # Evaluation metrics
│       │   ├── format_check.py        # Format validation
│       │   ├── executability.py       # Static executability check
│       │   ├── dynamic_executability.py  # Dynamic API call check
│       │   └── diversity.py           # Diversity metrics
│       ├── scripts/
│       │   └── run_full_test.py       # Unified entry point
│       └── results/                   # Evaluation outputs
│
└── Vision_Language_Data/              # Vision-Language Data
    ├── video_text_eval/               # Video-Text Evaluation
    │   ├── data_types.py              # VideoTextSample definition
    │   ├── loaders.py                 # Data loaders
    │   ├── metrics/                   # Evaluation metrics
    │   │   ├── frame_diversity.py     # Frame diversity (optical flow)
    │   │   ├── semantic_diversity.py  # Semantic diversity (Inception V3)
    │   │   ├── object_consistency.py  # Object consistency (CLIP)
    │   │   ├── cross_modal_consistency.py  # Cross-modal consistency (ViCLIP)
    │   │   ├── safety_bench.py        # Safety evaluation (GPT-4 Vision)
    │   │   ├── holistic_fidelity.py   # Holistic fidelity (VBench)
    │   │   ├── internvid/             # ViCLIP model
    │   │   └── vbench/                # VBench framework
    │   ├── scripts/
    │   │   └── run_full_test.py       # Unified entry point
    │   └── results/                   # Evaluation outputs
    │
    └── image_text_eval/               # Image-Text Evaluation
        ├── data_types.py              # ImageTextSample definition
        ├── loaders.py                 # Data loaders
        ├── image_executor.py          # Format checker base class
        ├── coco_executor.py           # COCO format checker
        ├── metrics/                   # Evaluation metrics
        │   ├── inception_score.py     # Inception Score
        │   ├── prompt_fidelity.py     # Prompt fidelity (CLIP)
        │   ├── well_formed_rate.py    # Well-formed rate
        │   └── c2pa_validation.py     # C2PA validation
        ├── scripts/
        │   └── run_full_test.py       # Unified entry point
        └── results/                   # Evaluation outputs
```

## 📊 Supported Datasets

| Domain | Dataset | Description |
|--------|---------|-------------|
| Math | LILA | Diverse mathematical reasoning |
| Math | OpenMathInstruct-1 | Large-scale math instructions |
| Agent | ToolBench | Tool calling dataset |
| Agent | xLAM-60k | API interaction dataset |
| Video-Text | General JSONL | Video-text pairs |
| Image-Text | COCO Caption | Image captioning dataset |

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

### Video-Text

| Metric | Description |
|--------|-------------|
| **Frame Diversity** | Optical flow based frame diversity |
| **Semantic Diversity** | Inception V3 feature diversity |
| **Object Consistency** | CLIP based object consistency |
| **Cross-Modal Consistency** | ViCLIP video-text alignment |
| **Safety Bench** | GPT-4 Vision safety evaluation |
| **Holistic Fidelity** | VBench comprehensive evaluation |

### Image-Text

| Metric | Description |
|--------|-------------|
| **Inception Score** | Image quality and diversity |
| **Prompt Fidelity** | CLIP image-text alignment |
| **Well-Formed Rate** | Data format correctness |
| **C2PA Validation** | Content provenance verification |

## 🚀 Quick Start

### Mathematical Reasoning Evaluation

```bash
cd Symbolic_and_Logical_Data/math_eval/scripts

# Format check
python run_full_test.py -d lila -m format_check

# Code validity
python run_full_test.py -d lila -m validity

# Diversity (KNN)
python run_full_test.py -d lila -m diversity --diversity-method knn
```

### API Agent Evaluation

```bash
cd Agent_Data/api_agent_eval/scripts

# Format check (parallel)
python run_full_test.py -d toolbench -m format_check --parallel

# Static executability (parallel)
python run_full_test.py -d toolbench -m executability --parallel

# Dynamic executability (requires RapidAPI Key)
export RAPIDAPI_KEY="your_key"
python run_full_test.py -d toolbench -m dynamic_executability
```

### Video-Text Evaluation

```bash
cd Vision_Language_Data/video_text_eval/scripts

# Frame diversity
python run_full_test.py -d test -m frame_diversity

# Semantic diversity
python run_full_test.py -d test -m semantic_diversity

# Cross-modal consistency
python run_full_test.py -d test -m cross_modal_consistency

# Run all metrics
python run_full_test.py -d test -m all
```

### Image-Text Evaluation

```bash
cd Vision_Language_Data/image_text_eval/scripts

# Inception Score
python run_full_test.py -d coco_caption -m inception_score

# Prompt fidelity
python run_full_test.py -d coco_caption -m prompt_fidelity

# Well-formed rate
python run_full_test.py -d coco_caption -m well_formed_rate

# Run all metrics
python run_full_test.py -d coco_caption -m all
```

## 🛠️ Architecture & Extensibility

This framework is designed with **extensibility** in mind. Each evaluation domain follows a consistent architecture pattern that allows users to easily add support for new datasets.

### Core Components

```
{domain}_eval/
├── data_types.py      # Fixed + Extensible data structure
├── loaders.py         # Base loader + dataset-specific loaders
├── {domain}_executor.py   # Base class for checkers
├── {dataset}_executor.py  # Dataset-specific implementations
├── metrics/           # Metric computation (fixed logic)
└── scripts/           # Entry points
```

### 1. Data Types: Fixed + Extensible Design

Each domain has a carefully designed data type that captures the **essential characteristics** of that data category:

```python
# Example: API Agent data type (api_agent_eval/data_types.py)
@dataclass
class APIAgentSample:
    # === Fixed Fields (represent core data value) ===
    query: str              # User query/instruction
    tools: List[Tool]       # Available tools/APIs
    conversations: List[Message]  # Interaction history
    
    # === Extensible Field (for dataset-specific needs) ===
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Design Philosophy:**
- **Fixed fields** represent the fundamental characteristics that define this data type's value
- **`metadata` field** allows storing dataset-specific fields that other datasets may not have

**Example Usage:**
```python
# ToolBench has additional fields like 'answer_generation'
sample = APIAgentSample(
    query="Search for weather",
    tools=[...],
    conversations=[...],
    metadata={
        "answer_generation": {...},  # ToolBench-specific
        "category": "weather",       # ToolBench-specific
    }
)
```

### 2. Executors: Base Class + Dataset Implementations

Each domain has a **base executor class** that defines the interface, and **dataset-specific implementations**.

**Why dataset-specific executors?**
- Different datasets have different validation requirements
- What needs to be checked varies by dataset (e.g., ToolBench needs API parameter validation, while xLAM needs function schema validation)
- Some datasets have unique fields that require custom checking logic

**When to use pre-built executors:**
- If your dataset is similar to existing ones (e.g., same format as ToolBench)
- If your validation needs are general enough

**When to write custom executors:**
- Your dataset has unique validation requirements
- You need to check dataset-specific fields stored in `metadata`

```python
# Base class (api_agent_eval/api_executor.py)
class FormatChecker(ABC):
    @abstractmethod
    def check(self, sample: APIAgentSample) -> Tuple[bool, List[str]]:
        """Check if sample has valid format. Returns (is_valid, error_messages)."""
        pass

class ExecutabilityChecker(ABC):
    @abstractmethod
    def check(self, sample: APIAgentSample) -> Tuple[bool, List[str]]:
        """Check if API calls are executable. Returns (is_valid, error_messages)."""
        pass
```

```python
# Dataset-specific implementation (api_agent_eval/toolbench_executor.py)
class ToolBenchFormatChecker(FormatChecker):
    def check(self, sample: APIAgentSample) -> Tuple[bool, List[str]]:
        errors = []
        # ToolBench-specific validation logic
        if not sample.query:
            errors.append("Missing query")
        if not sample.tools:
            errors.append("Missing tools")
        # ... more checks
        return len(errors) == 0, errors
```

### 3. Adding Support for a New Dataset

To evaluate a new dataset, you can either:
- **Use existing loaders/executors** if your dataset format is similar to supported ones
- **Create custom loaders/executors** if your dataset has unique requirements

#### Step 1: Create a Loader (in `loaders.py`) - *Optional*

The loader's main purpose is to **align your dataset fields with the evaluation data type**. 

**If your dataset uses standard field names** (e.g., `query`, `tools`, `video`, `text`, `image_path`, `caption`), you can directly use the pre-defined `GeneralLoader`:

```python
# No custom loader needed - just use GeneralLoader
from loaders import GeneralLoader
loader = GeneralLoader('/path/to/your_dataset.jsonl')
```

**If your dataset has different field names or requires custom parsing**, write a custom loader:

```python
# Add to loaders.py
from data_types import APIAgentSample

class MyDatasetLoader(BaseLoader):
    """
    Loader for MyDataset.
    Maps dataset-specific fields to APIAgentSample.
    """
    def iterate(self) -> Iterator[APIAgentSample]:
        with open(self.data_path) as f:
            for line in f:
                data = json.loads(line)
                # Map your dataset fields to standard fields
                yield APIAgentSample(
                    query=data['instruction'],      # your field -> standard field
                    tools=self._parse_tools(data['functions']),
                    conversations=self._parse_conversations(data['messages']),
                    metadata={
                        'custom_field': data.get('custom_field'),  # dataset-specific
                    }
                )
```

**When to use `GeneralLoader`:**
- Your dataset uses standard field names (e.g., `query`, `tools`, `video`, `text`)
- No special parsing is needed

**When to write a custom loader:**
- Your dataset has different field names (e.g., `instruction` instead of `query`)
- You need custom parsing logic (e.g., converting tool format)
- You have dataset-specific fields to store in `metadata`

#### Step 2: Create Dataset-Specific Executor

```python
# my_dataset_executor.py
from api_executor import FormatChecker, ExecutabilityChecker

class MyDatasetFormatChecker(FormatChecker):
    def check(self, sample: APIAgentSample) -> Tuple[bool, List[str]]:
        errors = []
        # Your dataset-specific validation logic
        if not sample.query:
            errors.append("Missing query")
        # ... custom checks for your dataset
        return len(errors) == 0, errors

class MyDatasetExecutabilityChecker(ExecutabilityChecker):
    def check(self, sample: APIAgentSample) -> Tuple[bool, List[str]]:
        # Your dataset-specific executability logic
        pass
```

#### Step 3: Register in run_full_test.py

```python
# Add to DATASETS config
DATASETS = {
    'my_dataset': {
        'name': 'My Dataset',
        'data_path': '/path/to/my_dataset.jsonl',
        'loader_class': MyDatasetLoader,
        'format_checker': MyDatasetFormatChecker,
        # ... other config
    },
}
```

#### Step 4: Run Evaluation

```bash
python run_full_test.py -d my_dataset -m format_check
python run_full_test.py -d my_dataset -m all
```

### 4. Metric Computation: Fixed Logic

The metric computation logic in `metrics/` is **fixed and reusable**. Once you provide the correct data loader and executor, metrics are computed automatically:

```python
# metrics/format_check.py - works for ANY dataset
def compute_format_check(
    data_iterator: Iterator[Sample],
    format_checker: FormatChecker,  # Your dataset-specific checker
    ...
) -> Dict[str, Any]:
    # Fixed computation logic
    for sample in data_iterator:
        is_valid, errors = format_checker.check(sample)
        # ... accumulate results
    return results
```

**Key Benefit:** You only need to implement dataset-specific loaders and executors. The metric computation, result saving, and reporting are handled automatically by the framework.

### Summary

| Component | Role | User Action |
|-----------|------|-------------|
| `data_types.py` | Define data structure | Use as-is (extend via `metadata`) |
| `loaders.py` | Load data into standard format | Implement for new datasets |
| `*_executor.py` | Base class for checkers | Inherit and implement |
| `metrics/` | Compute metrics | Use as-is (no changes needed) |
| `scripts/` | Entry points | Add dataset config |

## 📦 Requirements

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

## 📄 License

MIT License
