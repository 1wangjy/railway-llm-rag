#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/data16T/wjy/anaconda3/envs/unsloth/bin/python"
NVIDIA_ROOT="/data16T/wjy/anaconda3/envs/unsloth/lib/python3.11/site-packages/nvidia"
TRAIN_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:${NVIDIA_ROOT}/cublas/lib:${NVIDIA_ROOT}/cuda_cupti/lib:${NVIDIA_ROOT}/cuda_nvrtc/lib:${NVIDIA_ROOT}/cuda_runtime/lib:${NVIDIA_ROOT}/cudnn/lib:${NVIDIA_ROOT}/cufft/lib:${NVIDIA_ROOT}/cufile/lib:${NVIDIA_ROOT}/curand/lib:${NVIDIA_ROOT}/cusolver/lib:${NVIDIA_ROOT}/cusparse/lib:${NVIDIA_ROOT}/cusparselt/lib:${NVIDIA_ROOT}/nccl/lib:${NVIDIA_ROOT}/nvjitlink/lib:${NVIDIA_ROOT}/nvtx/lib"
BACKGROUND=0
case "${1:-}" in
  --background|-b)
    BACKGROUND=1
    shift
    ;;
  --help|-h)
    cat <<'EOF'
Usage:
  bash run_train.sh <CONFIG_FILE> [extra bootstrap_on.py arguments]
  bash run_train.sh --background <CONFIG_FILE> [extra bootstrap_on.py arguments]

Options:
  -b, --background  Start with nohup, then print the PID and launcher log path.
  -h, --help        Show this help message.
EOF
    exit 0
    ;;
esac

if [[ $# -eq 0 ]]; then
  echo "A training config is required; no default baseline is used." >&2
  echo "Usage: bash run_train.sh [--background] <CONFIG_FILE>" >&2
  exit 2
fi
CONFIG_FILE="$1"
shift

cd "${ROOT_DIR}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Training config not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [[ "${BACKGROUND}" -eq 1 ]]; then
  LAUNCHER_LOG_DIR="${ROOT_DIR}/launcher_logs"
  mkdir -p "${LAUNCHER_LOG_DIR}"
  CONFIG_NAME="$(basename "${CONFIG_FILE}" .py)"
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LAUNCHER_LOG="${LAUNCHER_LOG_DIR}/${TIMESTAMP}__${CONFIG_NAME}.nohup.log"
  PID_FILE="${LAUNCHER_LOG_DIR}/${TIMESTAMP}__${CONFIG_NAME}.pid"

  nohup bash "${ROOT_DIR}/run_train.sh" "${CONFIG_FILE}" "$@" \
    >"${LAUNCHER_LOG}" 2>&1 </dev/null &
  TRAIN_PID=$!
  printf '%s\n' "${TRAIN_PID}" >"${PID_FILE}"

  echo "Training started in background."
  echo "PID: ${TRAIN_PID}"
  echo "PID file: ${PID_FILE}"
  echo "Launcher log: ${LAUNCHER_LOG}"
  echo "Experiment log: $(dirname "${CONFIG_FILE}")/logs/"
  exit 0
fi

echo "Training framework: ${ROOT_DIR}/bootstrap_on.py"
echo "Training config: ${ROOT_DIR}/${CONFIG_FILE}"

exec env \
  LD_LIBRARY_PATH="${TRAIN_LD_LIBRARY_PATH}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" -u bootstrap_on.py --config "${CONFIG_FILE}" "$@"
