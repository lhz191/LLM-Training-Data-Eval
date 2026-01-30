#!/bin/bash
#SBATCH --job-name=diversity_knn
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/petrelfs/liuhaoze/main/Symbolic_and_Logical_Data/math_eval/results/lila/diversity_knn_all-MiniLM-L6-v2_%j.log

cd /mnt/petrelfs/liuhaoze/main/Symbolic_and_Logical_Data/math_eval

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

echo "=========================================="
echo "Diversity KNN 评估 (GPU)"
echo "=========================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

echo ">>> 运行 Diversity KNN 评估..."
python3 -u scripts/run_full_test.py --metric diversity --dataset lila --diversity-method knn --embedding-model all-MiniLM-L6-v2

echo ""
echo "=========================================="
echo "结束时间: $(date)"
