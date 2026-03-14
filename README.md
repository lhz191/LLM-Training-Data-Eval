<div align="center">

# 🔬 LLM Training Data Evaluation

**A comprehensive framework for evaluating LLM training data quality before used for training**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README.md) | [中文](README_CN.md)

</div>

---

## 📖 Overview

This framework provides systematic evaluation metrics for LLM training data quality across multiple data modalities:

- **Symbolic & Logical Data** - Mathematical reasoning, code, formal logic
- **Agent Data** - Tool-calling, API interaction, web navigation
- **Vision-Language Data** - Image-text, video-text multimodal data
- **Text Data** - Pure text corpora
- **Tabular Data** - Structured table data
- **Semi-Structured & Graph Data** - Knowledge graphs, semi-structured data

## Two-Layer Evaluation

This framework addresses a critical gap in LLM data pipelines: **there is no standard way to assess data quality between generation and training** before actually training. We provide two complementary evaluation layers:

```
                        LLM generates data
                              |
                              v
                ┌─────────────────────────┐
                │  Layer 1: Data Audit    │  task-agnostic
                │  "Can this data be used │  validity, fidelity,
                │   at all?"              │  diversity, safety
                └────────────┬────────────┘
                             |
                    filter out defective data
                             |
                             v
                ┌─────────────────────────┐
                │  Layer 2: Task-Specific │  per-task metrics
                │  "Is this data good for │  pass@k, exec rate,
                │   task X?"              │  VBench, BLEU, ...
                └────────────┬────────────┘
                             |
                             v
                      post-training (SFT/DPO/RLHF)
```

**Layer 1 — Data Quality Audit** evaluates intrinsic properties of each data point independent of downstream usage. It catches deterministic defects: incorrect math solutions, non-executable code, malformed JSON, unsafe content. These checks are cheap, fast, and apply universally.

**Layer 2 — Task-Specific Evaluation** measures data fitness for a particular post-training objective using domain-appropriate metrics and executors. Different datasets require different checkers (e.g., ToolBench needs API parameter validation, LILA needs symbolic execution, COCO needs caption format compliance).

Both layers are organized by data modality and share the same loader/executor infrastructure. For each modality, `metrics/` provides data-level audit functions, while `{domain}_eval/` provides task-specific evaluation pipelines.

### Relationship Between Layers

The two layers are not independent — they share the same evaluation dimensions (validity, fidelity, diversity, safety) but operate at different granularities:

```
Single data point        Dataset level             Training level
  (correctness,     (+ diversity, fidelity,    (+ actually train a model,
   safety, ...)       format compliance, ...)    measure PGR, ...)

|---- Layer 1 ----|---- Layer 2 -----------|     out of scope
```

**Layer 2 is a superset of Layer 1.** Task-specific evaluation naturally includes data-point quality checks, plus additional dataset-level statistics and format compliance requirements. The degree of overlap between layers varies by modality:

| Modality | Layer 1 (Data Audit) | Layer 2 (Task-Specific) | Overlap |
|----------|---------------------|------------------------|---------|
| **Graph / Tabular** | Structural validity, MMD, diversity | Node classification, link prediction, ML efficacy | Low — genuinely different |
| **Text** | Grammar, toxicity, perplexity, dedup | Instruction difficulty, response quality for SFT, preference data quality | Low to medium |
| **Multimodal** | Visual quality (IS, VBench), safety, CLIP score | + VQA correctness, captioning (BLEU/CIDEr), retrieval, multimodal SFT quality | Medium — current metrics cover generation tasks; understanding tasks need Layer 2 |
| **Agent / Reasoning** | Executability, answer correctness, code execution | Same checkers used in both layers | High — verification inherently requires domain-specific execution |

<details>
<summary><b>Example: Why Graph data needs both layers</b></summary>

Suppose an LLM generates a knowledge graph triple: `(Einstein, born_in, Physics)`

**Layer 1 (Data Audit)** checks structural properties only:
- Is it a valid graph? (head, relation, tail all present) -> Pass
- Degree distribution matches reference graphs? -> Pass
- Not a duplicate of other generated triples? -> Pass

**Layer 2 (Task-Specific, e.g. KG Completion)** checks semantic correctness:
- Is the fact correct? -> Fail (Einstein was born in Ulm, not "Physics")
- Type constraint: `born_in` tail should be a location, not a discipline -> Fail

Layer 1 says this data is structurally fine. Layer 2 says it is semantically wrong and will degrade a knowledge graph completion model. This gap is why Graph and Tabular modalities benefit most from two-layer evaluation — structural validity and semantic correctness are fundamentally different concerns.

</details>

> **Why not just use Layer 2?** Layer 1 operates on individual data points immediately after generation, before data is organized into task-specific datasets. It serves as a fast pre-filter with no assumptions about downstream usage. Layer 2 requires knowing the target task and dataset schema, and evaluates at the collection level.
>
> **Context.** Recent work (AgoraBench, ACL 2025) shows that intrinsic data features can partially predict downstream training effectiveness (R² = 0.325 via PCA on soft metrics like LLM judge scores). Our framework focuses on **hard verification** — deterministic checks like code execution and answer comparison — which catches errors that soft metrics miss entirely.

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
│       ├── scripts/
│       │   └── run_full_test.py       # Unified entry point
│       └── results/                   # Evaluation outputs
│
├── Agent_Data/                        # Agent Data
│   └── api_agent_eval/
│       ├── data_types.py              # Data type definitions
│       ├── loaders.py                 # Data loaders
│       ├── api_executor.py            # Executor base class
│       ├── toolbench_executor.py      # ToolBench executor
│       ├── xlam_executor.py           # xLAM executor
│       ├── metrics/                   # Evaluation metrics
│       ├── scripts/
│       │   └── run_full_test.py       # Unified entry point
│       └── results/                   # Evaluation outputs
│
├── Vision_Language_Data/              # Vision-Language Data
│   ├── video_text_eval/               # Video-Text Evaluation
│   │   ├── data_types.py              # VideoTextSample definition
│   │   ├── loaders.py                 # Data loaders
│   │   ├── metrics/                   # Evaluation metrics
│   │   ├── scripts/
│   │   │   └── run_full_test.py       # Unified entry point
│   │   └── results/                   # Evaluation outputs
│   │
│   └── image_text_eval/               # Image-Text Evaluation
│       ├── data_types.py              # ImageTextSample definition
│       ├── loaders.py                 # Data loaders
│       ├── image_executor.py          # Format checker base class
│       ├── coco_executor.py           # COCO format checker
│       ├── metrics/                   # Evaluation metrics
│       ├── scripts/
│       │   └── run_full_test.py       # Unified entry point
│       └── results/                   # Evaluation outputs
│
├── Text_Data/                         # Text Data (coming soon)
│
├── Tabular_Data/                      # Tabular Data (coming soon)
│
└── Semi_Structured_Graph_Data/        # Semi-Structured & Graph Data (coming soon)
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
