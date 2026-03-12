# Win Rate 评估预处理指南

本目录包含 Win Rate 评估的预处理工具，包括生成 caption、创建标注数据文件、浏览器标注界面等。

最终的标注结果保存在 `evaluation_results/` 目录中，由 `metrics/win_rate.py` 进行计算。

## 工作流程

```
1. 生成 Caption (generate_chameleon_captions.py)
   ↓
2. 创建标注数据 (create_annotation_data 函数)
   ↓
3. 浏览器标注 (annotation_server.py)
   ↓
4. 保存到 evaluation_results/
   ↓
5. 计算 Win Rate (metrics/win_rate.py)
```

## 快速开始

### 步骤 1: 安装依赖

```bash
pip install flask
```

如果需要使用 Chameleon，参考 [Chameleon GitHub](https://github.com/facebookresearch/chameleon)：

```bash
pip install -U git+https://github.com/facebookresearch/chameleon.git
```

### 步骤 2: 生成 Caption

使用 Chameleon 或其他模型生成 caption：

```bash
cd /mnt/shared-storage-user/liurun/LLMDataBenchmark

python -m Multimodal.data_utils.win_rate.generate_chameleon_captions \
    --dataset Datasets/Multimodal/coco_caption.json \
    --output results/chameleon_7b_generation.json \
    --dataset-root Datasets/Multimodal \
    --model-size 7b \
    --max-samples 100
```

**输出格式**：

```json
{
  "model": "chameleon_7b",
  "dataset": "Datasets/Multimodal/coco_caption.json",
  "results": [
    {
      "id": 48,
      "image_id": 318556,
      "image_path": "train2014/COCO_train2014_000000318556.jpg",
      "generated": "A clean bathroom with modern fixtures"
    }
  ]
}
```

### 步骤 3: 创建标注数据文件

创建标注数据文件用于浏览器标注：

```bash
cd /mnt/shared-storage-user/liurun/LLMDataBenchmark

python -c "
from Multimodal.data_utils.win_rate.annotation_server import create_annotation_data
create_annotation_data(
    generation_file='results/chameleon_7b_generation.json',
    ground_truth_file='Datasets/Multimodal/coco_caption.json',
    output_file='evaluation_results/chameleon_7b_annotation.json',
    dataset_root='Datasets/Multimodal'
)
"
```

或者创建一个简单的脚本：

```python
# create_annotation.py
from Multimodal.data_utils.win_rate.annotation_server import create_annotation_data

create_annotation_data(
    generation_file='results/chameleon_7b_generation.json',
    ground_truth_file='Datasets/Multimodal/coco_caption.json',
    output_file='evaluation_results/chameleon_7b_annotation.json',
    dataset_root='Datasets/Multimodal'
)
```

```bash
python create_annotation.py
```

### 步骤 4: 启动标注服务器

启动浏览器标注界面：

```bash
cd /mnt/shared-storage-user/liurun/LLMDataBenchmark

python -m Multimodal.data_utils.win_rate.annotation_server \
    --annotation-file evaluation_results/chameleon_7b_annotation.json \
    --dataset-root Datasets/Multimodal \
    --port 5000
```

### 步骤 5: 在浏览器中标注

1. 打开浏览器访问：`http://localhost:5000`
2. 查看每个样本的图片、生成的 caption 和 ground truth
3. 点击按钮标注：
   - **Win**: 生成的 caption 更好
   - **Tie**: 两者相当
   - **Loss**: ground truth 更好
4. 使用键盘快捷键：
   - `←` / `→`: 切换样本
   - `1`: Win
   - `2`: Tie
   - `3`: Loss
5. 标注会自动保存到 `evaluation_results/chameleon_7b_annotation.json`

### 步骤 6: 计算 Win Rate

标注完成后，使用 `metrics/win_rate.py` 计算 Win Rate：

```bash
cd /mnt/shared-storage-user/liurun/LLMDataBenchmark

# 方法 1: 指定文件路径
python -m Multimodal.metrics.win_rate \
    --file evaluation_results/chameleon_7b_annotation.json

# 方法 2: 使用模型名称（会自动查找 evaluation_results/{model}_annotation.json）
python -m Multimodal.metrics.win_rate --model chameleon_7b
```

**输出示例**：

```
Win Rate Results:
  Model: chameleon_7b
  File: evaluation_results/chameleon_7b_annotation.json
  Win Rate: 0.6500 (65.00%)

Statistics:
  Wins: 50
  Ties: 30
  Losses: 20
  Total: 100
```

## 完整示例

```bash
# 0. 进入项目根目录
cd /mnt/shared-storage-user/liurun/LLMDataBenchmark

# 1. 生成 caption
python -m Multimodal.data_utils.win_rate.generate_chameleon_captions \
    --dataset Datasets/Multimodal/coco_caption.json \
    --output results/chameleon_7b_generation.json \
    --dataset-root Datasets/Multimodal \
    --model-size 7b \
    --max-samples 100

# 2. 创建标注数据文件（使用 Python 脚本）
python -c "
from Multimodal.data_utils.win_rate.annotation_server import create_annotation_data
create_annotation_data(
    'results/chameleon_7b_generation.json',
    'Datasets/Multimodal/coco_caption.json',
    'evaluation_results/chameleon_7b_annotation.json',
    'Datasets/Multimodal'
)
"

# 3. 启动标注服务器（在终端保持运行）
python -m Multimodal.data_utils.win_rate.annotation_server \
    --annotation-file evaluation_results/chameleon_7b_annotation.json \
    --dataset-root Datasets/Multimodal

# 4. 在浏览器打开 http://localhost:5000 进行标注

# 5. 标注完成后，计算 Win Rate（在新终端运行）
cd /mnt/shared-storage-user/liurun/LLMDataBenchmark
python -m Multimodal.metrics.win_rate --model chameleon_7b
```

## 文件说明

- **generate_chameleon_captions.py**: 使用 Chameleon 模型生成 caption
- **annotation_server.py**: 浏览器标注服务器和标注数据创建工具
- **README_WIN_RATE.md**: 本文档

## 输出文件位置

所有标注结果文件保存在 `Multimodal/evaluation_results/` 目录中，格式为：

```
evaluation_results/
  ├── chameleon_7b_annotation.json
  ├── chameleon_30b_annotation.json
  └── ...
```

这些文件由 `metrics/win_rate.py` 读取并计算 Win Rate。

## 注意事项

1. **工作目录**: 所有命令必须在项目根目录（`LLMDataBenchmark/`）运行
2. **图片路径**: 确保 `--dataset-root` 参数正确，图片文件可以正常访问
3. **自动保存**: 标注结果会自动保存，刷新页面不会丢失
4. **文件位置**: 标注结果文件必须保存在 `evaluation_results/` 目录中
5. **模型名称**: 使用 `--model` 参数时，会自动查找 `evaluation_results/{model}_annotation.json`

## 常见问题

### ModuleNotFoundError: No module named 'Multimodal'

**解决方法**: 确保在项目根目录运行命令
```bash
cd /mnt/shared-storage-user/liurun/LLMDataBenchmark
python -m Multimodal.data_utils.win_rate.generate_chameleon_captions --help
```

### 图片无法显示

**解决方法**: 检查 `--dataset-root` 参数是否正确，确保图片文件存在
```bash
ls Datasets/Multimodal/train2014/COCO_train2014_*.jpg | head -1
```

### 文件不存在错误

**解决方法**: 确保标注结果文件保存在 `evaluation_results/` 目录中
```bash
ls evaluation_results/*.json
```
