#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_SCRIPT="${ROOT_DIR}/infer_qwen25_7b.py"
PYTHON_BIN="/data16T/wjy/anaconda3/envs/unsloth/bin/python"
PYTORCH_NVIDIA_LIB_ROOT="/data16T/wjy/anaconda3/envs/unsloth/lib/python3.11/site-packages/nvidia"
CLEAN_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:${PYTORCH_NVIDIA_LIB_ROOT}/cublas/lib:${PYTORCH_NVIDIA_LIB_ROOT}/cuda_cupti/lib:${PYTORCH_NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${PYTORCH_NVIDIA_LIB_ROOT}/cuda_runtime/lib:${PYTORCH_NVIDIA_LIB_ROOT}/cudnn/lib:${PYTORCH_NVIDIA_LIB_ROOT}/cufft/lib:${PYTORCH_NVIDIA_LIB_ROOT}/cufile/lib:${PYTORCH_NVIDIA_LIB_ROOT}/curand/lib:${PYTORCH_NVIDIA_LIB_ROOT}/cusolver/lib:${PYTORCH_NVIDIA_LIB_ROOT}/cusparse/lib:${PYTORCH_NVIDIA_LIB_ROOT}/cusparselt/lib:${PYTORCH_NVIDIA_LIB_ROOT}/nccl/lib:${PYTORCH_NVIDIA_LIB_ROOT}/nvjitlink/lib:${PYTORCH_NVIDIA_LIB_ROOT}/nvtx/lib"

if [[ ! -f "${INFER_SCRIPT}" ]]; then
  echo "infer script not found: ${INFER_SCRIPT}" >&2
  exit 1
fi

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${ROOT_DIR}/logs/infer"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}__infer_qwen25_7b.log"
mkdir -p "${LOG_DIR}"

echo "parameter_file: ${ROOT_DIR}/configs/qwen25_7b_config.py"
echo "log_file: ${LOG_FILE}"

if [[ "${1:-}" == "--fg" ]]; then
  shift
  env LD_LIBRARY_PATH="${CLEAN_LD_LIBRARY_PATH}" "${PYTHON_BIN}" "${INFER_SCRIPT}" "$@" 2>&1 | tee "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

nohup env LD_LIBRARY_PATH="${CLEAN_LD_LIBRARY_PATH}" "${PYTHON_BIN}" "${INFER_SCRIPT}" "$@" > "${LOG_FILE}" 2>&1 &
echo "pid: $!"
