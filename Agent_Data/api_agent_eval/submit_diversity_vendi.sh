#!/bin/bash
#SBATCH --job-name=diversity_vendi
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval/results/toolbench/diversity_vendi_all-MiniLM-L6-v2_%j.log

cd /mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

echo "=========================================="
echo "Diversity Vendi Score 评估 (all-MiniLM-L6-v2)"
echo "=========================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

echo ">>> 运行 Vendi Score 评估 (all-MiniLM-L6-v2)..."
python3 -u run_full_test.py --metric diversity --dataset toolbench --diversity-method vendi --embedding-model all-MiniLM-L6-v2 --vendi-batch-size 6500 --num-gpus 8

echo ""
echo "=========================================="
echo "结束时间: $(date)"
