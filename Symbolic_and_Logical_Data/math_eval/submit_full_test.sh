#!/bin/bash
#SBATCH --job-name=lila_validity
#SBATCH --partition=t-cpu-new
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/petrelfs/liuhaoze/main/Symbolic_and_Logical_Data/math_eval/results/lila/validity_%j.log

cd /mnt/petrelfs/liuhaoze/main/Symbolic_and_Logical_Data/math_eval

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

echo "=========================================="
echo "LILA 数据集 Validity 评估"
echo "=========================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "CPU 核心数: $SLURM_CPUS_PER_TASK"
echo ""

echo ">>> 运行 LILA Validity 评估 (32 进程并行)..."
python3 -u run_full_test.py --metric validity --dataset lila --parallel --workers 32

echo ""
echo "=========================================="
echo "结束时间: $(date)"
