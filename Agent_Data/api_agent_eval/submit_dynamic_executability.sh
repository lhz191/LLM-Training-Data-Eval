#!/bin/bash
#SBATCH --job-name=dynamic_exec
#SBATCH --partition=t-cpu-new
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval/results/toolbench/dynamic_executability_%j.log

cd /mnt/petrelfs/liuhaoze/main/Agent_Data/api_agent_eval

# 初始化 conda 和代理
source ~/.bashrc
proxy_on

# RapidAPI Key
export RAPIDAPI_KEY='5759f99c03msh28bf16c50c393e8p15e166jsn63f572088a56'

echo "=========================================="
echo "Dynamic Executability 评估"
echo "=========================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "CPU 核心数: $SLURM_CPUS_PER_TASK"
echo ""

# 使用 run_full_test.py 运行
# --max-samples 1000: 采样 1000 个样本
# --workers 16: 16 个并行线程
# --timeout 180: API 调用超时 180 秒（3分钟）
python3 -u run_full_test.py --dataset toolbench --metric dynamic_executability --max-samples 1000 --workers 16 --timeout 180

echo ""
echo "=========================================="
echo "结束时间: $(date)"

