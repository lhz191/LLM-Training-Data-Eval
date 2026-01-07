#!/bin/bash
#SBATCH --job-name=diversity_vendi_mpnet
#SBATCH -p TDS
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/petrelfs/liuhaoze/main/Symbolic_and_Logical_Data/math_eval/results/openmath/diversity_vendi_all-mpnet-base-v2_%j.log

cd /mnt/petrelfs/liuhaoze/main/Symbolic_and_Logical_Data/math_eval

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

echo "=========================================="
echo "Diversity Vendi Score 评估 (all-mpnet-base-v2) - OpenMath"
echo "=========================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

echo ">>> 运行 Vendi Score 评估 (all-mpnet-base-v2)..."
# all-mpnet-base-v2: 768 维，需要更多显存，使用多 GPU 并行
python3 -u run_full_test.py --metric diversity --dataset openmathinstruct --diversity-method vendi --embedding-model all-mpnet-base-v2 --vendi-batch-size 6500 --num-gpus 8
echo ""

echo "=========================================="
echo "结束时间: $(date)"
