#!/bin/bash
#SBATCH --job-name=diversity_knn_xlam
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval/results/xlam/diversity_knn_Qwen3-Embedding-8B_%j.log

cd /mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

# 限制 OpenBLAS 线程数，防止 KNN 计算时崩溃
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

echo "=========================================="
echo "Diversity KNN 评估 (Qwen3-Embedding-8B) - xLAM"
echo "=========================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

echo ">>> 运行 Diversity KNN 评估..."
python3 -u scripts/run_full_test.py --metric diversity --dataset xlam --diversity-method knn --embedding-model Qwen/Qwen3-Embedding-8B --embedding-batch-size 4

echo ""
echo "=========================================="
echo "结束时间: $(date)"
