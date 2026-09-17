#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${ABLATION_CUDA_VISIBLE_DEVICES:-0}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_generation_lengths.py" "$@"
