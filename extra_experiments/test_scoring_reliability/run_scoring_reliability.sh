#!/usr/bin/env bash
# =============================================================================
# 评分复测信度实验 - 一键启动脚本
#
# 用法（在项目根目录下执行）：
#   bash extra_experiments/test_scoring_reliability/run_scoring_reliability.sh
#
# 选项：
#   --skip-mfa    跳过 MFA 对齐，使用已有 TextGrid（MFA 未安装时使用）
#   --runs N      指定重复测量次数（默认 5）
#
# 兼容：
#   - WSL（Linux 路径，如 /mnt/d/...）
#   - Windows 原生 bash（Git Bash / MSYS2）
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 用户可修改的配置
# ---------------------------------------------------------------------------

# 【重要】项目根目录：若自动检测失败（常见于 CRLF 问题），在此手动填写 WSL 路径
# 例如：MANUAL_PROJECT_ROOT="/mnt/d/workdir"
# 留空则自动检测
MANUAL_PROJECT_ROOT=""

# Python 解释器：留空则自动检测
MANUAL_PYTHON_BIN=""

# 重复测量次数
MANUAL_RUNS="5"

# 是否跳过 MFA 对齐（1=跳过，0=重跑 MFA）
# 如果你没有安装 MFA，或只想测试 GFCC+DTW 评分的确定性，设为 1
MANUAL_SKIP_MFA="0"

# ---------------------------------------------------------------------------
# 路径解析（兼容 CRLF、WSL、Git Bash）
# ---------------------------------------------------------------------------
# 去除 BASH_SOURCE 可能携带的 \r（CRLF 文件在 WSL 下的常见问题）
_raw_source="${BASH_SOURCE[0]}"
_raw_source="${_raw_source//$'\r'/}"
SCRIPT_DIR="$(cd "$(dirname "${_raw_source}")" && pwd)"

if [[ -n "${MANUAL_PROJECT_ROOT}" ]]; then
    PROJECT_ROOT="${MANUAL_PROJECT_ROOT%$'\r'}"   # 同样去 \r
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

if [[ -z "${PROJECT_ROOT}" ]]; then
    echo "[ERROR] 无法自动检测项目根目录。" >&2
    echo "[ERROR] 请在脚本顶部手动填写 MANUAL_PROJECT_ROOT，例如：" >&2
    echo "[ERROR]   MANUAL_PROJECT_ROOT=\"/mnt/d/workdir\"" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 自动检测 Python
# ---------------------------------------------------------------------------
if [[ -n "${MANUAL_PYTHON_BIN}" ]]; then
    PYTHON_BIN="${MANUAL_PYTHON_BIN}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "[ERROR] 未找到 Python，请安装 Python 3.8+ 或设置 MANUAL_PYTHON_BIN" >&2
    exit 1
fi

PYTHON_VERSION=$("${PYTHON_BIN}" --version 2>&1 | head -1)
echo "[INFO] 使用 Python: ${PYTHON_BIN} (${PYTHON_VERSION})"

# ---------------------------------------------------------------------------
# 参数检测（支持命令行覆盖）
# ---------------------------------------------------------------------------
SKIP_MFA_FLAG=""
RUNS="${MANUAL_RUNS}"
for arg in "$@"; do
    case "${arg}" in
        --skip-mfa) SKIP_MFA_FLAG="--skip-mfa" ;;
        --runs=*) RUNS="${arg#--runs=}" ;;
        --runs) shift; RUNS="${1:-5}" ;;
    esac
done

if [[ "${MANUAL_SKIP_MFA}" == "1" && -z "${SKIP_MFA_FLAG}" ]]; then
    SKIP_MFA_FLAG="--skip-mfa"
fi

# ---------------------------------------------------------------------------
# 检查必要文件
# ---------------------------------------------------------------------------
echo "[INFO] 项目根目录: ${PROJECT_ROOT}"

REFERENCE_CORPUS="${PROJECT_ROOT}/data/fijian/raw_corpus"
REFERENCE_ALIGNED="${PROJECT_ROOT}/data/fijian/aligned/reference"
LEARNER_PCM="${PROJECT_ROOT}/data/fijian/learner_pcm"
DICT_PATH="${PROJECT_ROOT}/data/fijian/fijian_bootstrap_arpa.dict"
PIPELINE_SCRIPT="${PROJECT_ROOT}/scripts/mfa_pipeline/run/batch_gfcc_dtw_eval.py"

for path in "${REFERENCE_CORPUS}" "${REFERENCE_ALIGNED}" "${LEARNER_PCM}" "${DICT_PATH}" "${PIPELINE_SCRIPT}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[ERROR] 必要路径不存在: ${path}" >&2
        exit 1
    fi
done

if [[ -z "${SKIP_MFA_FLAG}" ]]; then
    if ! command -v mfa >/dev/null 2>&1; then
        echo "[WARNING] 未找到 MFA（mfa 命令不在 PATH 中）"
        echo "[WARNING] 自动切换为 --skip-mfa 模式（使用已有 TextGrid）"
        SKIP_MFA_FLAG="--skip-mfa"
    else
        echo "[INFO] MFA 已找到: $(which mfa)"
    fi
fi

# ---------------------------------------------------------------------------
# 安装依赖（可选，如需要）
# ---------------------------------------------------------------------------
check_package() {
    "${PYTHON_BIN}" -c "import ${1}" 2>/dev/null && return 0 || return 1
}

MISSING_PKGS=""
for pkg in numpy pandas scipy matplotlib spafe fastdtw librosa praatio soundfile; do
    if ! check_package "${pkg}"; then
        MISSING_PKGS="${MISSING_PKGS} ${pkg}"
    fi
done

if [[ -n "${MISSING_PKGS}" ]]; then
    echo "[INFO] 检测到缺少 Python 包: ${MISSING_PKGS}"
    echo "[INFO] 尝试自动安装..."
    "${PYTHON_BIN}" -m pip install --quiet numpy pandas scipy matplotlib spafe fastdtw librosa praatio soundfile
fi

# ---------------------------------------------------------------------------
# Step 1: 运行主实验
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Step 1: 运行评分复测信度主实验"
echo "  重复次数: ${RUNS}  跳过MFA: ${SKIP_MFA_FLAG:-否}"
echo "============================================================"

"${PYTHON_BIN}" "${SCRIPT_DIR}/scoring_reliability_experiment.py" \
    --runs "${RUNS}" \
    ${SKIP_MFA_FLAG}

# ---------------------------------------------------------------------------
# Step 2: 分析结果
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Step 2: 计算 ICC(2,1) 与信度指标"
echo "============================================================"

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_scoring_reliability.py"

# ---------------------------------------------------------------------------
# 完成
# ---------------------------------------------------------------------------
RESULTS_DIR="${SCRIPT_DIR}/results"
echo ""
echo "============================================================"
echo "  [DONE] 实验完成"
echo "============================================================"
echo "  抽样结果:   ${RESULTS_DIR}/selected_pairs.csv"
echo "  合并评分:   ${RESULTS_DIR}/all_scores.csv"
echo "  ICC 汇总:   ${RESULTS_DIR}/icc_analysis/icc_report.csv"
echo "  Markdown:   ${RESULTS_DIR}/icc_analysis/report.md"
echo "  图表目录:   ${RESULTS_DIR}/icc_analysis/plots/"
echo "============================================================"
