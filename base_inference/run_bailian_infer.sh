#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data16T/wjy/anaconda3/envs/unsloth/bin/python}"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "缺少 DASHSCOPE_API_KEY。请先执行：" >&2
  echo '  read -s -p "请输入百炼 API Key: " DASHSCOPE_API_KEY; echo; export DASHSCOPE_API_KEY' >&2
  exit 2
fi

# Use the Unsloth environment's packaged CUDA libraries even when the active
# shell is `(base)`. Do not derive this from CONDA_PREFIX: in a base shell that
# points at /data16T/wjy/anaconda3 and omits libcusparseLt.so.0.
UNSLOTH_ENV_ROOT="/data16T/wjy/anaconda3/envs/unsloth"
NVIDIA_LIBRARY_ROOT="${UNSLOTH_ENV_ROOT}/lib/python3.11/site-packages/nvidia"
CUDA_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:${UNSLOTH_ENV_ROOT}/lib:${NVIDIA_LIBRARY_ROOT}/cublas/lib:${NVIDIA_LIBRARY_ROOT}/cuda_cupti/lib:${NVIDIA_LIBRARY_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIBRARY_ROOT}/cuda_runtime/lib:${NVIDIA_LIBRARY_ROOT}/cudnn/lib:${NVIDIA_LIBRARY_ROOT}/cufft/lib:${NVIDIA_LIBRARY_ROOT}/cufile/lib:${NVIDIA_LIBRARY_ROOT}/curand/lib:${NVIDIA_LIBRARY_ROOT}/cusolver/lib:${NVIDIA_LIBRARY_ROOT}/cusparse/lib:${NVIDIA_LIBRARY_ROOT}/cusparselt/lib:${NVIDIA_LIBRARY_ROOT}/nccl/lib:${NVIDIA_LIBRARY_ROOT}/nvjitlink/lib:${NVIDIA_LIBRARY_ROOT}/nvtx/lib"

cd "${SCRIPT_DIR}"
exec env \
  LD_LIBRARY_PATH="${CUDA_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" -u run_bailian_infer.py "$@"
