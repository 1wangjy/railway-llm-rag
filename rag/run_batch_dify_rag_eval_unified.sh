#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/batch_dify_rag_eval_unified.py" "$@"
fi
if [[ -z "${DIFY_API_KEY:-}" ]]; then
  echo 'DIFY_API_KEY is required. Export it before running this script.' >&2
  exit 2
fi

exec env PYTHONUNBUFFERED=1   "$PYTHON_BIN" -u "$SCRIPT_DIR/batch_dify_rag_eval_unified.py" "$@"
