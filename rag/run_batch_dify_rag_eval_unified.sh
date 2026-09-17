#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/data16T/wjy/anaconda3/envs/unsloth/bin/python"
NVIDIA_ROOT="/data16T/wjy/anaconda3/envs/unsloth/lib/python3.11/site-packages/nvidia"
CUDA_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:${NVIDIA_ROOT}/cublas/lib:${NVIDIA_ROOT}/cuda_cupti/lib:${NVIDIA_ROOT}/cuda_nvrtc/lib:${NVIDIA_ROOT}/cuda_runtime/lib:${NVIDIA_ROOT}/cudnn/lib:${NVIDIA_ROOT}/cufft/lib:${NVIDIA_ROOT}/cufile/lib:${NVIDIA_ROOT}/curand/lib:${NVIDIA_ROOT}/cusolver/lib:${NVIDIA_ROOT}/cusparse/lib:${NVIDIA_ROOT}/cusparselt/lib:${NVIDIA_ROOT}/nccl/lib:${NVIDIA_ROOT}/nvjitlink/lib:${NVIDIA_ROOT}/nvtx/lib"
API_KEY_FILE="${ROOT_DIR}/rag_eval/.dify_api_key"

if [[ -z "${DIFY_API_KEY:-}" && -r "${API_KEY_FILE}" ]]; then
  IFS= read -r DIFY_API_KEY < "${API_KEY_FILE}"
  export DIFY_API_KEY
fi

cd "${ROOT_DIR}"
exec env \
  LD_LIBRARY_PATH="${CUDA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" -u rag_eval/batch_dify_rag_eval_unified.py "$@"
