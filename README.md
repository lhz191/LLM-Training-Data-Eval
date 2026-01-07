# Data Quality Evaluation Framework

数据质量评估框架，用于评估数学推理数据和 API Agent 数据的质量。

## 项目结构

```
├── Symbolic_and_Logical_Data/     # 数学推理数据评估
│   └── math_eval/                 # 评估代码
│       ├── data_types.py          # 数据类型定义
│       ├── loaders.py             # 数据加载器
│       ├── code_executor.py       # 代码执行基类
│       ├── lila_executor.py       # LILA 数据集执行器
│       ├── openmath_executor.py   # OpenMath 数据集执行器
│       ├── format_check.py        # 格式检查指标
│       ├── validity.py            # 代码有效性指标
│       ├── reasoning_validity.py  # 推理有效性指标
│       ├── faithfulness.py        # 忠实性指标
│       ├── diversity.py           # 多样性指标 (Vendi Score / KNN)
│       └── run_full_test.py       # 统一入口
│
└── Agent_Data/                    # API Agent 数据评估
    └── api_agent_eval/            # 评估代码
        ├── data_types.py          # 数据类型定义
        ├── loaders.py             # 数据加载器 (ToolBench, xLAM)
        ├── api_executor.py        # 执行器基类
        ├── toolbench_executor.py  # ToolBench 执行器
        ├── xlam_executor.py       # xLAM 执行器
        ├── format_check.py        # 格式检查指标
        ├── executability.py       # 静态可执行性指标
        ├── dynamic_executability.py  # 动态可执行性指标
        ├── diversity.py           # 多样性指标 (Vendi Score / KNN)
        └── run_full_test.py       # 统一入口
```

## 支持的数据集

### 数学推理 (Symbolic_and_Logical_Data)
- **LILA**: 多样化数学推理数据集
- **OpenMathInstruct-1**: 大规模数学指令数据集
- **NuminaMath-CoT**: 链式推理数学数据集

### API Agent (Agent_Data)
- **ToolBench**: 工具调用数据集
- **xLAM-60k**: API 调用数据集

## 评估指标

### 数学推理指标
| 指标 | 描述 |
|------|------|
| Format Check | 格式正确性检查 |
| Validity | 代码可执行性和正确性 |
| Reasoning Validity | 推理过程有效性 |
| Faithfulness | 答案与推理的一致性 |
| Diversity | 数据多样性 (Vendi Score / KNN) |

### API Agent 指标
| 指标 | 描述 |
|------|------|
| Format Check | 格式正确性检查 |
| Executability | 静态可执行性 (API 定义、参数匹配) |
| Dynamic Executability | 动态可执行性 (实际 API 调用) |
| Diversity | 数据多样性 (Vendi Score / KNN) |

## 使用方法

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

# 格式检查
python run_full_test.py -d toolbench -m format_check --parallel

# 静态可执行性
python run_full_test.py -d toolbench -m executability --parallel

# 动态可执行性 (需要 RapidAPI Key)
export RAPIDAPI_KEY="your_key"
python run_full_test.py -d toolbench -m dynamic_executability

# 多样性 (KNN)
python run_full_test.py -d toolbench -m diversity --diversity-method knn

# 多样性 (Vendi Score)
python run_full_test.py -d toolbench -m diversity --diversity-method vendi
```

## Slurm 提交

每个模块都提供了 Slurm 提交脚本：

```bash
# 数学推理
sbatch submit_full_test.sh
sbatch submit_diversity_gpu.sh

# API Agent
sbatch submit_format_check.sh
sbatch submit_executability.sh
sbatch submit_diversity_vendi.sh
sbatch submit_diversity_knn.sh
```

## 依赖

```
torch
transformers
sentence-transformers
numpy
tqdm
```

## License

MIT

