#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/data16T/wjy/anaconda3/envs/unsloth/bin/python"
NVIDIA_LIB_ROOT="/data16T/wjy/anaconda3/envs/unsloth/lib/python3.11/site-packages/nvidia"
CUDA_LIBS="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda-12.4/lib64:${NVIDIA_LIB_ROOT}/cublas/lib:${NVIDIA_LIB_ROOT}/cuda_cupti/lib:${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib:${NVIDIA_LIB_ROOT}/cuda_runtime/lib:${NVIDIA_LIB_ROOT}/cudnn/lib:${NVIDIA_LIB_ROOT}/cufft/lib:${NVIDIA_LIB_ROOT}/cufile/lib:${NVIDIA_LIB_ROOT}/curand/lib:${NVIDIA_LIB_ROOT}/cusolver/lib:${NVIDIA_LIB_ROOT}/cusparse/lib:${NVIDIA_LIB_ROOT}/cusparselt/lib:${NVIDIA_LIB_ROOT}/nccl/lib:${NVIDIA_LIB_ROOT}/nvjitlink/lib:${NVIDIA_LIB_ROOT}/nvtx/lib"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${ABLATION_CUDA_VISIBLE_DEVICES:-2}"
export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/evaluate_generation_lengths.py" "$@"
