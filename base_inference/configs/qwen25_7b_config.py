"""Qwen2.5-7B pure-inference parameters.

Only edit this file when changing the Qwen2.5 inference experiment.
The shared inference framework lives in ``infer_common.py``.
"""

import os
from types import SimpleNamespace


# ============== 1. Paths and experiment identity ==========================
MODEL_PATH = os.path.join(os.getenv("MODEL_ROOT", "models"), "Qwen2.5-7B-Instruct")
EVAL_DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "data/testdata799.json")
OUTPUT_PATH = "outputs/qwen25_7b_base_testdata799_system_max1024_predictions.json"
CUDA_VISIBLE_DEVICES = "2"


# ============== 2. Dataset fields =========================================
DATASET_SPLIT = "train"
DATASET_BATCHED = True
INSTRUCTION_COLUMN = "instruction"
INPUT_COLUMN = "input"
OUTPUT_COLUMN = "output"
SYSTEM_COLUMN = "system"
MAX_SAMPLES = None  # None means all evaluation samples.


# ============== 3. Inference and generation ===============================
MAX_SEQ_LENGTH = 2048
GENERATION_MAX_NEW_TOKENS = 1024
PROGRESS_EVERY = 10
DO_SAMPLE = False
TEMPERATURE = None
TOP_P = None
TOP_K = None
ADD_GENERATION_PROMPT = True
SKIP_SPECIAL_TOKENS = True
CLEAN_UP_TOKENIZATION_SPACES = False


# ============== 4. Model loading and quantization =========================
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
UNSLOTH_DTYPE = None
LOCAL_FILES_ONLY = True
TRUST_REMOTE_CODE = True
TOKENIZER_USE_FAST = True
TORCH_DTYPE = "float16"
DEVICE_MAP = "auto"
BNB_COMPUTE_DTYPE = "float16"
BNB_USE_DOUBLE_QUANT = True
BNB_QUANT_TYPE = "nf4"


# ============== 5. Evaluation =============================================
ENABLE_BLEU = True
BLEU_MAX_ORDER = 4
BLEU_SMOOTH_VALUE = 0.0  # Set to 0.1 if add-k smoothing is desired.

ENABLE_BERTSCORE = True
BERTSCORE_MODEL_TYPE = os.getenv("BERTSCORE_MODEL", "bert-base-chinese")
BERTSCORE_NUM_LAYERS = 12
BERTSCORE_LANG = "zh"
BERTSCORE_BATCH_SIZE = 8
ENABLE_SEGMENTED_BERTSCORE = True
SEGMENTED_BERTSCORE_CHUNK_TOKENS = 450


# ============== 6. Result serialization ===================================
OUTPUT_ENSURE_ASCII = False
OUTPUT_INDENT = 2


# Fixed model metadata. Usually no change is needed below this line.
CONFIG = SimpleNamespace(
    key="qwen25_7b",
    description="Evaluate base Qwen2.5-7B-Instruct.",
    model_path=MODEL_PATH,
    default_cuda_visible_devices=CUDA_VISIBLE_DEVICES,
    output_path=OUTPUT_PATH,
    model_family="standard",
    eval_dataset_path=EVAL_DATASET_PATH,
    dataset_split=DATASET_SPLIT,
    dataset_batched=DATASET_BATCHED,
    instruction_column=INSTRUCTION_COLUMN,
    input_column=INPUT_COLUMN,
    output_column=OUTPUT_COLUMN,
    system_column=SYSTEM_COLUMN,
    max_samples=MAX_SAMPLES,
    max_seq_length=MAX_SEQ_LENGTH,
    generation_max_new_tokens=GENERATION_MAX_NEW_TOKENS,
    progress_every=PROGRESS_EVERY,
    do_sample=DO_SAMPLE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    add_generation_prompt=ADD_GENERATION_PROMPT,
    skip_special_tokens=SKIP_SPECIAL_TOKENS,
    clean_up_tokenization_spaces=CLEAN_UP_TOKENIZATION_SPACES,
    load_in_4bit=LOAD_IN_4BIT,
    load_in_8bit=LOAD_IN_8BIT,
    unsloth_dtype=UNSLOTH_DTYPE,
    local_files_only=LOCAL_FILES_ONLY,
    trust_remote_code=TRUST_REMOTE_CODE,
    tokenizer_use_fast=TOKENIZER_USE_FAST,
    torch_dtype=TORCH_DTYPE,
    device_map=DEVICE_MAP,
    bnb_compute_dtype=BNB_COMPUTE_DTYPE,
    bnb_use_double_quant=BNB_USE_DOUBLE_QUANT,
    bnb_quant_type=BNB_QUANT_TYPE,
    chatglm_use_cache=False,
    enable_bleu=ENABLE_BLEU,
    bleu_max_order=BLEU_MAX_ORDER,
    bleu_smooth_value=BLEU_SMOOTH_VALUE,
    enable_bertscore=ENABLE_BERTSCORE,
    bertscore_model_type=BERTSCORE_MODEL_TYPE,
    bertscore_num_layers=BERTSCORE_NUM_LAYERS,
    bertscore_lang=BERTSCORE_LANG,
    bertscore_batch_size=BERTSCORE_BATCH_SIZE,
    enable_segmented_bertscore=ENABLE_SEGMENTED_BERTSCORE,
    segmented_bertscore_chunk_tokens=SEGMENTED_BERTSCORE_CHUNK_TOKENS,
    output_ensure_ascii=OUTPUT_ENSURE_ASCII,
    output_indent=OUTPUT_INDENT,
)
