#!/bin/bash
#==============================================================================
# submit_eval.sh - 通用评测任务提交脚本
#
# 使用方式:
#   ./submit_eval.sh <modality> <dataset> <metric> [extra_args...]
#
# 示例:
#   ./submit_eval.sh gui mind2web diversity
#   ./submit_eval.sh gui mind2web diversity --parallel        # 并行加速
#   ./submit_eval.sh gui mind2web diversity --max-samples 100
#   ./submit_eval.sh api toolbench diversity --diversity-method knn
#   ./submit_eval.sh math lila validity
#
# 环境变量 (可选):
#   PARTITION     - 分区名 (默认: TDS)
#   TIME_LIMIT    - 时间限制 (默认: 4:00:00)
#   NUM_GPUS      - GPU 数量 (默认: 1)
#   CPUS          - CPU 核数 (默认: 16)
#   CONDA_ENV     - Conda 环境 (默认: webshop)
#==============================================================================

set -e

# ============ 参数解析 ============
if [ $# -lt 3 ]; then
    echo "Usage: $0 <modality> <dataset> <metric> [extra_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 gui mind2web diversity"
    echo "  $0 gui mind2web diversity --max-samples 100"
    echo "  $0 api toolbench format_check"
    echo ""
    echo "Environment variables:"
    echo "  PARTITION=TDS TIME_LIMIT=2:00:00 NUM_GPUS=8 $0 gui mind2web diversity"
    exit 1
fi

MODALITY=$1
DATASET=$2
METRIC=$3
shift 3
EXTRA_ARGS="$@"

# ============ 默认配置 ============
PARTITION=${PARTITION:-TDS}
TIME_LIMIT=${TIME_LIMIT:-4:00:00}
NUM_GPUS=${NUM_GPUS:-1}
CPUS=${CPUS:-16}
CONDA_ENV=${CONDA_ENV:-base}

# ============ 路径配置 ============
PROJECT_ROOT="/mnt/petrelfs/liuhaoze/main_new"

# 根据模态确定结果目录
case ${MODALITY} in
    api)
        RESULTS_DIR="${PROJECT_ROOT}/modalities/Agent_Data/api_agent_eval/results/${DATASET}"
        ;;
    gui)
        RESULTS_DIR="${PROJECT_ROOT}/modalities/Agent_Data/text_gui_agent_eval/results/${DATASET}"
        ;;
    math)
        RESULTS_DIR="${PROJECT_ROOT}/modalities/Symbolic_and_Logical_Data/math_eval/results/${DATASET}"
        ;;
    report)
        RESULTS_DIR="${PROJECT_ROOT}/modalities/Multimodal_Data/image_to_report_eval/results/${DATASET}"
        ;;
    image)
        RESULTS_DIR="${PROJECT_ROOT}/modalities/Vision_Language_Data/image_text_eval/results/${DATASET}"
        ;;
    video)
        RESULTS_DIR="${PROJECT_ROOT}/modalities/Vision_Language_Data/video_text_eval/results/${DATASET}"
        ;;
    *)
        RESULTS_DIR="${PROJECT_ROOT}/results/${MODALITY}/${DATASET}"
        ;;
esac

LOG_DIR="${RESULTS_DIR}"

# 创建日志目录
mkdir -p "${LOG_DIR}"

# ============ 作业名称 ============
JOB_NAME="${MODALITY}_${DATASET}_${METRIC}"

# ============ 生成 SBATCH 脚本 ============
SBATCH_SCRIPT=$(mktemp /tmp/sbatch_eval_XXXXXX.sh)

# GPU 行：NUM_GPUS=0 时不加 --gres（兼容 CPU-only 分区）
if [ "${NUM_GPUS}" -gt 0 ] 2>/dev/null; then
    GPU_LINE="#SBATCH --gres=gpu:${NUM_GPUS}"
else
    GPU_LINE=""
fi

cat > "${SBATCH_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH -p ${PARTITION}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
${GPU_LINE}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOG_DIR}/${METRIC}_results_%j.log

# ============ 环境初始化 ============
cd ${PROJECT_ROOT}

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# 限制线程数，防止数值计算时崩溃
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

# ============ 打印信息 ============
echo "=========================================="
echo "LLM Training Data Evaluation"
echo "=========================================="
echo "作业名称: ${JOB_NAME}"
echo "模态: ${MODALITY}"
echo "数据集: ${DATASET}"
echo "指标: ${METRIC}"
echo "额外参数: ${EXTRA_ARGS}"
echo ""
echo "开始时间: \$(date)"
echo "节点: \$(hostname)"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo "=========================================="
echo ""

# ============ 运行评测 ============
python3 -u evaluate.py ${MODALITY} ${DATASET} ${METRIC} ${EXTRA_ARGS}

echo ""
echo "=========================================="
echo "结束时间: \$(date)"
echo "=========================================="
EOF

# ============ 提交作业 ============
echo "=========================================="
echo "提交评测作业"
echo "=========================================="
echo "  模态:    ${MODALITY}"
echo "  数据集:  ${DATASET}"
echo "  指标:    ${METRIC}"
echo "  额外参数: ${EXTRA_ARGS:-无}"
echo ""
echo "  分区:    ${PARTITION}"
echo "  GPU:     ${NUM_GPUS}"
echo "  CPU:     ${CPUS}"
echo "  时间:    ${TIME_LIMIT}"
echo "  日志:    ${LOG_DIR}/${METRIC}_results_<jobid>.log"
echo "=========================================="
echo ""

# 提交并显示结果
sbatch "${SBATCH_SCRIPT}"

# 清理临时文件
rm -f "${SBATCH_SCRIPT}"

echo ""
echo "使用 'squeue -u \$(whoami)' 查看作业状态"

