"""集中管理多模型推理与评估参数。"""

import os
from dataclasses import dataclass

from configs.qwen25_7b_config import CONFIG as QWEN25_7B_CONFIG
from configs.qwen3_8b_config import CONFIG as QWEN3_8B_CONFIG
from configs.ling3_tiny_config import CONFIG as LING3_TINY_CONFIG


# ============== 1、推荐参数区：通常只需要改这里 ==========================

# 模型与输出路径
MODEL_ROOTS = {
    "chatglm3_6b": "/data16T/wjy/models/chatglm3-6b",
    "deepseek_llm_7b_chat": "/data16T/wjy/models/deepseek-llm-7b-chat",
    "deepseek_r1_7b": "/data16T/wjy/models/DeepSeek-R1-Distill-Qwen-7B",
    "llama31_8b": "/data16T/wjy/models/Llama-3.1-8B-Instruct",
    "qwen25_7b": "/data16T/wjy/models/Qwen2.5-7B-Instruct",
    "qwen3_8b": "/data16T/wjy/models/Qwen3-8B",
    "ling3_tiny": "/data16T/wjy/models/Ling-3.0-tiny",
}

MODEL_GPUS = {
    "chatglm3_6b": "0",
    "deepseek_llm_7b_chat": "3",
    "deepseek_r1_7b": "3",
    "llama31_8b": "4",
    "qwen25_7b": "2",
    "qwen3_8b": "3",
    "ling3_tiny": "3",
}

OUTPUT_ROOT = "outputs"

# 数据路径与字段
eval_dataset_path = "/data16T/wjy/Learn_llm/铁路大模型/dataset_splits/testdata799.json"
dataset_split = "train"
dataset_batched = True
instruction_column = "instruction"
input_column = "input"
output_column = "output"
system_column = "system"

# 推理主参数
max_samples = None
max_seq_length = 2048
generation_max_new_tokens = 1024
progress_every = 10

# 模型加载与量化
load_in_4bit = True
load_in_8bit = False
unsloth_dtype = None
local_files_only = True
trust_remote_code = True
tokenizer_use_fast = True
torch_dtype = "float16"
device_map = "auto"
bnb_compute_dtype = "float16"
bnb_use_double_quant = True
bnb_quant_type = "nf4"
chatglm_use_cache = False

# 生成策略
do_sample = False
temperature = None
top_p = None
top_k = None
add_generation_prompt = True
skip_special_tokens = True
clean_up_tokenization_spaces = False

# BLEU 配置
enable_bleu = True
bleu_max_order = 4
bleu_smooth_value = 0.0  # 如需 add-k 平滑可改为 0.1

# BERTScore 配置
enable_bertscore = True
bertscore_model_type = "/data16T/wjy/models/Bert"
bertscore_num_layers = 12
bertscore_lang = "zh"
bertscore_batch_size = 8
enable_segmented_bertscore = True
segmented_bertscore_chunk_tokens = 450

# 结果文件格式
output_ensure_ascii = False
output_indent = 2


# ============== 2、模型元数据：新增模型时在这里补一项 ====================

model_settings = {
    "chatglm3_6b": {
        "description": "Evaluate base ChatGLM3-6B on testdata799.json.",
        "output_file": "chatglm3_6b_base_testdata799_system_max1024_predictions.json",
        "model_family": "chatglm",
    },
    "deepseek_llm_7b_chat": {
        "description": "Evaluate base DeepSeek-LLM-7B-Chat on testdata799.json.",
        "output_file": "deepseek_llm_7b_chat_base_testdata799_system_max1024_predictions.json",
        "model_family": "standard",
    },
    "deepseek_r1_7b": {
        "description": "Evaluate base DeepSeek-R1-Distill-Qwen-7B on testdata799.json.",
        "output_file": "deepseek_r1_distill_qwen_7b_base_testdata799_system_max1024_predictions.json",
        "model_family": "standard",
    },
    "llama31_8b": {
        "description": "Evaluate base Llama-3.1-8B-Instruct on testdata799.json.",
        "output_file": "llama31_8b_instruct_base_testdata799_system_max1024_predictions.json",
        "model_family": "standard",
    },
    "qwen25_7b": {
        "description": "Evaluate base Qwen2.5-7B-Instruct on testdata799.json.",
        "output_file": "qwen25_7b_base_testdata799_system_max1024_predictions.json",
        "model_family": "standard",
    },
    "qwen3_8b": {
        "description": "Evaluate base Qwen3-8B on testdata799.json.",
        "output_file": "qwen3_8b_base_testdata799_system_max1024_predictions.json",
        "model_family": "standard",
    },
    "ling3_tiny": {
        "description": "Evaluate Ling-3.0-tiny with reasoning disabled on testdata799.json.",
        "output_file": "ling3_tiny_base_testdata799_system_max1024_no_thinking_predictions.json",
        "model_family": "standard",
    },
}


# ============== 3、内部配置组装：通常不需要修改 ==========================

@dataclass(frozen=True)
class InferenceConfig:
    key: str
    description: str
    model_path: str
    default_cuda_visible_devices: str
    output_path: str
    model_family: str


model_configs = {
    key: InferenceConfig(
        key=key,
        description=settings["description"],
        model_path=MODEL_ROOTS[key],
        default_cuda_visible_devices=MODEL_GPUS[key],
        output_path=os.path.join(OUTPUT_ROOT, settings["output_file"]),
        model_family=settings["model_family"],
    )
    for key, settings in model_settings.items()
}

# Preserve ``python run_infer.py --model qwen25_7b`` while sourcing all
# Qwen2.5 parameters from its independent configuration file.
model_configs["qwen25_7b"] = QWEN25_7B_CONFIG
model_configs["qwen3_8b"] = QWEN3_8B_CONFIG
model_configs["ling3_tiny"] = LING3_TINY_CONFIG


def get_config(key: str) -> InferenceConfig:
    return model_configs[key]
