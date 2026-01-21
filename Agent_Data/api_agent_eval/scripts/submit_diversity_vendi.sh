#!/bin/bash
#SBATCH --job-name=diversity_vendi_xlam
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval/results/xlam/diversity_vendi_Qwen3-Embedding-8B_%j.log

cd /mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

echo "=========================================="
echo "Diversity Vendi Score 评估 (Qwen3-Embedding-8B) - xLAM"
echo "=========================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

echo ">>> 运行 Vendi Score 评估 (Qwen3-Embedding-8B)..."
python3 -u scripts/run_full_test.py --metric diversity --dataset xlam --diversity-method vendi --embedding-model Qwen/Qwen3-Embedding-8B --vendi-batch-size 6500 --num-gpus 8 --embedding-batch-size 4

echo ""
echo "=========================================="
echo "结束时间: $(date)"
