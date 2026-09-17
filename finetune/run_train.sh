#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKGROUND=0

case "${1:-}" in
  --background|-b) BACKGROUND=1; shift ;;
  --help|-h)
    cat <<'EOF'
Usage: bash finetune/run_train.sh [--background] finetune/train_config_final.py

Required environment variables normally include MODEL_ROOT, TRAIN_DATASET_PATH,
EVAL_DATASET_PATH and BERTSCORE_MODEL. Activate the Python environment that
contains Unsloth, PyTorch and the dependencies in requirements.txt first.
EOF
    exit 0 ;;
esac

if [[ $# -eq 0 ]]; then
  echo 'A training config is required. See --help.' >&2
  exit 2
fi

CONFIG_FILE="$1"
shift
if [[ "$CONFIG_FILE" != /* ]]; then
  CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Training config not found: $CONFIG_FILE" >&2
  exit 1
fi

if [[ "$BACKGROUND" -eq 1 ]]; then
  LOG_DIR="$ROOT_DIR/launcher_logs"
  mkdir -p "$LOG_DIR"
  NAME="$(basename "$CONFIG_FILE" .py)"
  STAMP="$(date +%Y%m%d_%H%M%S)"
  nohup bash "$ROOT_DIR/run_train.sh" "$CONFIG_FILE" "$@"     >"$LOG_DIR/${STAMP}__${NAME}.nohup.log" 2>&1 </dev/null &
  echo "Training started: PID $!"
  exit 0
fi

exec env PYTHONUNBUFFERED=1   "$PYTHON_BIN" -u "$ROOT_DIR/bootstrap_on.py" --config "$CONFIG_FILE" "$@"
