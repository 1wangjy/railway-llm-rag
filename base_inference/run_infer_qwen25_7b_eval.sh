#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="$SCRIPT_DIR/logs/infer"
mkdir -p "$LOG_DIR"

if [[ "${1:-}" == "--fg" ]]; then
  shift
  exec "$PYTHON_BIN" "$SCRIPT_DIR/infer_qwen25_7b.py" "$@"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
nohup "$PYTHON_BIN" "$SCRIPT_DIR/infer_qwen25_7b.py" "$@"   >"$LOG_DIR/${STAMP}__infer_qwen25_7b.log" 2>&1 < /dev/null &
echo "Inference started: PID $!"
