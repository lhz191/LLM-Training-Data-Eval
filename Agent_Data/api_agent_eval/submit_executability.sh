#!/bin/bash
#SBATCH --job-name=executability_toolbench
#SBATCH --partition=t-cpu-new
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval/results/toolbench/executability_%j.log

cd /mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate base

echo "=========================================="
echo "Executability 评估"
echo "=========================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "CPU 核心数: $SLURM_CPUS_PER_TASK"
echo ""

# 使用 run_full_test.py 运行
# 默认 toolbench，如需 xlam 改 --dataset xlam 和 output log 路径
python3 -u run_full_test.py --dataset toolbench --metric executability --parallel --workers 32

echo ""
echo "=========================================="
echo "结束时间: $(date)"

