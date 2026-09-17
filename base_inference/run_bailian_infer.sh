#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo 'DASHSCOPE_API_KEY is required. Export it before running this script.' >&2
  exit 2
fi

cd "$SCRIPT_DIR"
exec env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u run_bailian_infer.py "$@"
