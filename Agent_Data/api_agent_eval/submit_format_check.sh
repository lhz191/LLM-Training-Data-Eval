#!/bin/bash
#SBATCH --job-name=format_check_toolbench
#SBATCH --partition=t-cpu-new
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval/results/toolbench/format_check_%j.log

cd /mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

echo "=========================================="
echo "Format Check 评估 (ToolBench)"
echo "=========================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "CPU 核心数: $SLURM_CPUS_PER_TASK"
echo ""

# ToolBench 数据集
python3 -u run_full_test.py --dataset toolbench --metric format_check --parallel --workers 32

echo ""
echo "=========================================="
echo "结束时间: $(date)"
