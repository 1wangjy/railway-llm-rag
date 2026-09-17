"""Bailian API inference configuration aligned with local baseline evaluation."""

import os

# API model and endpoint. The shared Beijing endpoint accepts API keys from
# workspaces in the same region; a workspace-dedicated endpoint can override it.
BAILIAN_MODEL = "glm-5"
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV = "DASHSCOPE_API_KEY"

# Dataset and output paths
EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "data/testdata799.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "outputs/glm5_direct_testdata799_system_max1024_predictions.json")

# Keep these fields identical to infer_config.py.
DATASET_SPLIT = "train"
INSTRUCTION_COLUMN = "instruction"
INPUT_COLUMN = "input"
OUTPUT_COLUMN = "output"
SYSTEM_COLUMN = "system"

# The existing second-round local baseline files were generated with user-only
# prompts. Keep Bailian on the same prompt protocol for direct comparison.
INCLUDE_SYSTEM_PROMPT = False

# Generation. Thinking is disabled so only the final answer is compared.
MAX_SAMPLES = None
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = None
ENABLE_THINKING = False
PROGRESS_EVERY = 10

# Request retry and incremental resume
REQUEST_TIMEOUT = 600
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
RESUME = True

# Metrics: identical to infer_config.py
BLEU_MAX_ORDER = 4
BLEU_SMOOTH_VALUE = 0.0
BERTSCORE_MODEL_TYPE = os.getenv("BERTSCORE_MODEL", "bert-base-chinese")
BERTSCORE_NUM_LAYERS = 12
BERTSCORE_LANG = "zh"
BERTSCORE_BATCH_SIZE = 8
SEGMENTED_BERTSCORE_CHUNK_TOKENS = 450

# BERTScore runs locally after all API responses have been collected.
DEFAULT_CUDA_VISIBLE_DEVICES = "4"

# Attributes consumed by the metric functions in infer_common.py.
local_files_only = True
tokenizer_use_fast = True
skip_special_tokens = True
clean_up_tokenization_spaces = False
output_ensure_ascii = False
output_indent = 2
