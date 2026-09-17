"""环境：unsloth"""
# export CUDA_VISIBLE_DEVICES=0
# python /data16T/wjy/Learn_llm/train_unsloth.py
# 训练结果保存在远程服务器的日志：20250818_1.log

# nohup python train_unsloth.py > 20260415_resume.log 2>&1 &

import argparse
import hashlib
import importlib
import importlib.util
import math
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from functools import lru_cache
from types import SimpleNamespace


TRAINING_FRAMEWORK_VERSION = "new_7b_segmented_bertscore_v2_20260823"

REQUIRED_CONFIG_KEYS = {
    "model_path", "bootstrap_adapter_dir", "output_root", "experiment_name",
    "train_dataset_path", "eval_dataset_path", "expected_train_samples",
    "expected_eval_samples", "cuda_visible_devices",
    "max_seq_length", "model_dtype", "load_in_4bit", "load_in_8bit",
    "full_finetuning", "local_files_only", "resume_enabled", "bootstrap_enabled",
    "lora_r", "lora_alpha", "lora_dropout", "lora_bias", "lora_target_modules",
    "use_gradient_checkpointing", "lora_random_state", "use_rslora", "loftq_config",
    "dataset_split", "dataset_batched", "instruction_column", "input_column",
    "output_column", "system_column", "eval_add_generation_prompt",
    "selection_eval_max_samples", "selection_eval_seed",
    "seed", "num_train_epochs", "learning_rate",
    "training_max_length", "per_device_train_batch_size",
    "gradient_accumulation_steps", "warmup_steps", "warmup_ratio",
    "expected_effective_batch_size", "expected_optimizer_steps",
    "completion_only_loss", "logging_steps", "optim",
    "weight_decay", "lr_scheduler_type", "report_to", "packing", "eval_strategy",
    "eval_steps", "save_strategy", "save_steps", "save_total_limit",
    "load_best_model_at_end", "metric_for_best_model", "greater_is_better",
    "early_stopping_patience", "early_stopping_min_delta",
    "generation_max_new_tokens", "generation_do_sample",
    "generation_skip_special_tokens", "generation_progress_every",
    "enable_bertscore", "bertscore_model_type",
    "bertscore_num_layers", "bertscore_lang", "bertscore_batch_size",
    "bertscore_local_files_only", "bertscore_tokenizer_use_fast",
    "bertscore_skip_special_tokens", "bertscore_clean_up_tokenization_spaces",
    "enable_segmented_bertscore", "segmented_bertscore_chunk_tokens",
}
OPTIONAL_CONFIG_DEFAULTS = {
    "early_stopping_enabled": True,
}

def load_experiment_config(config_reference):
    if config_reference.endswith(".py") or os.path.sep in config_reference:
        config_path = os.path.abspath(config_reference)
        spec = importlib.util.spec_from_file_location("selected_train_config", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load training config: {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(config_reference)
        config_path = getattr(module, "__file__", config_reference)

    raw_config = getattr(module, "CONFIG", None)
    if not isinstance(raw_config, dict):
        raise TypeError(f"Training config must export a CONFIG dict: {config_reference}")

    missing_keys = sorted(REQUIRED_CONFIG_KEYS - raw_config.keys())
    if missing_keys:
        raise ValueError(f"Training config is missing keys: {', '.join(missing_keys)}")
    unknown_keys = sorted(
        raw_config.keys() - REQUIRED_CONFIG_KEYS - OPTIONAL_CONFIG_DEFAULTS.keys()
    )
    if unknown_keys:
        raise ValueError(f"Training config contains unknown keys: {', '.join(unknown_keys)}")
    effective_config = {**OPTIONAL_CONFIG_DEFAULTS, **raw_config}
    return SimpleNamespace(**effective_config), os.path.abspath(config_path), raw_config


class TeeStream:
    """Write console output to both the terminal and the experiment log."""

    def __init__(self, terminal_stream, log_stream):
        self.terminal_stream = terminal_stream
        self.log_stream = log_stream

    def write(self, data):
        self.terminal_stream.write(data)
        self.log_stream.write(data)
        return len(data)

    def flush(self):
        self.terminal_stream.flush()
        self.log_stream.flush()

    def isatty(self):
        return self.terminal_stream.isatty()

    def fileno(self):
        return self.terminal_stream.fileno()


def enable_experiment_logging(config_path):
    config_directory = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    log_directory = os.path.join(config_directory, "logs")
    os.makedirs(log_directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_directory, f"{timestamp}__{config_name}.log")
    log_stream = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_stream)
    sys.stderr = TeeStream(sys.__stderr__, log_stream)
    return log_path


config_parser = argparse.ArgumentParser(
    description="Qwen2.5-7B fixed LoRA training framework with pluggable experiment configs."
)
config_parser.add_argument(
    "--config",
    required=True,
    help="Python module or .py path exporting a CONFIG dict.",
)
config_parser.add_argument(
    "--print-config",
    action="store_true",
    help="Validate and print the selected config without importing training libraries.",
)
config_args = config_parser.parse_args()
cfg, selected_config_path, selected_config_dict = load_experiment_config(config_args.config)

if config_args.print_config:
    print(f"config_file = {selected_config_path}")
    print(f"log_directory = {os.path.join(os.path.dirname(selected_config_path), 'logs')}")
    print(json.dumps(selected_config_dict, ensure_ascii=False, indent=2))
    raise SystemExit(0)

experiment_log_path = enable_experiment_logging(selected_config_path)
print(f"Training log: {experiment_log_path}")

# 1. 强制按物理 PCI 总线顺序排序（这就和 nvidia-smi 看到的一样了）
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# 2. 指定使用第 0 号卡
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices

"""unsloth依赖一定要放最前面"""
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import ExportableState, TrainerCallback
from peft import set_peft_model_state_dict
import torch

try:
    from safetensors.torch import load_file as safe_load_file
except ImportError:
    safe_load_file = None

try:
    from bert_score import score as bert_score_score
except ImportError:
    bert_score_score = None


def tokenize_for_rouge(text):
    """Tokenize Chinese ROUGE/BLEU inputs consistently at character level."""
    return [char for char in text.strip() if not char.isspace()]


def rouge_n_f1(prediction_tokens, reference_tokens, n):
    if len(prediction_tokens) < n or len(reference_tokens) < n:
        return 0.0

    prediction_ngrams = Counter(
        tuple(prediction_tokens[i:i + n]) for i in range(len(prediction_tokens) - n + 1)
    )
    reference_ngrams = Counter(
        tuple(reference_tokens[i:i + n]) for i in range(len(reference_tokens) - n + 1)
    )
    overlap = sum((prediction_ngrams & reference_ngrams).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(prediction_ngrams.values())
    recall = overlap / sum(reference_ngrams.values())
    return 2 * precision * recall / (precision + recall)


def lcs_length(prediction_tokens, reference_tokens):
    if not prediction_tokens or not reference_tokens:
        return 0

    dp = [0] * (len(reference_tokens) + 1)
    for prediction_token in prediction_tokens:
        previous = 0
        for index, reference_token in enumerate(reference_tokens, start=1):
            current = dp[index]
            if prediction_token == reference_token:
                dp[index] = previous + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            previous = current
    return dp[-1]


def rouge_l_f1(prediction_tokens, reference_tokens):
    if not prediction_tokens or not reference_tokens:
        return 0.0

    lcs = lcs_length(prediction_tokens, reference_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_average_rouge(predictions, references):
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for prediction, reference in zip(predictions, references):
        prediction_tokens = tokenize_for_rouge(prediction)
        reference_tokens = tokenize_for_rouge(reference)
        rouge1_scores.append(rouge_n_f1(prediction_tokens, reference_tokens, 1))
        rouge2_scores.append(rouge_n_f1(prediction_tokens, reference_tokens, 2))
        rougeL_scores.append(rouge_l_f1(prediction_tokens, reference_tokens))

    num_samples = max(len(predictions), 1)
    return {
        "rouge1": sum(rouge1_scores) / num_samples * 100,
        "rouge2": sum(rouge2_scores) / num_samples * 100,
        "rougeL": sum(rougeL_scores) / num_samples * 100,
    }


def compute_corpus_bleu4(predictions, references):
    """Compute single-reference corpus BLEU-4 with clipped n-gram precision."""
    if not predictions or not references:
        return 0.0

    max_order = 4
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    prediction_length = 0
    reference_length = 0

    for prediction, reference in zip(predictions, references):
        prediction_tokens = tokenize_for_rouge(prediction)
        reference_tokens = tokenize_for_rouge(reference)
        prediction_length += len(prediction_tokens)
        reference_length += len(reference_tokens)

        for order in range(1, max_order + 1):
            prediction_ngrams = Counter(
                tuple(prediction_tokens[index:index + order])
                for index in range(max(len(prediction_tokens) - order + 1, 0))
            )
            reference_ngrams = Counter(
                tuple(reference_tokens[index:index + order])
                for index in range(max(len(reference_tokens) - order + 1, 0))
            )
            matches_by_order[order - 1] += sum(
                (prediction_ngrams & reference_ngrams).values()
            )
            possible_matches_by_order[order - 1] += sum(prediction_ngrams.values())

    if prediction_length == 0:
        return 0.0

    precisions = []
    for matches, possible_matches in zip(matches_by_order, possible_matches_by_order):
        if possible_matches == 0:
            continue
        precision = matches / possible_matches
        if precision == 0:
            return 0.0
        precisions.append(precision)

    if not precisions:
        return 0.0

    geometric_mean = math.exp(
        sum(math.log(precision) for precision in precisions) / len(precisions)
    )
    brevity_penalty = (
        1.0
        if prediction_length >= reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    return brevity_penalty * geometric_mean * 100


@lru_cache(maxsize=4)
def load_bertscore_tokenizer_and_limit(model_type):
    if not model_type:
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(
        model_type,
        local_files_only=cfg.bertscore_local_files_only,
        use_fast=cfg.bertscore_tokenizer_use_fast,
    )
    config = AutoConfig.from_pretrained(
        model_type,
        local_files_only=cfg.bertscore_local_files_only,
    )

    max_positions = getattr(config, "max_position_embeddings", None)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if tokenizer_limit is not None and tokenizer_limit > 100000:
        tokenizer_limit = None

    raw_limit = max_positions if max_positions is not None else tokenizer_limit
    if raw_limit is None:
        return tokenizer, None

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_limit = max(int(raw_limit) - int(special_tokens), 1)
    return tokenizer, content_limit


def truncate_texts_for_bertscore(texts, tokenizer, max_content_tokens):
    if tokenizer is None or max_content_tokens is None:
        return texts, 0, 0

    truncated_texts = []
    truncated_count = 0
    longest_original_length = 0

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_length = len(token_ids)
        longest_original_length = max(longest_original_length, token_length)
        if token_length > max_content_tokens:
            token_ids = token_ids[:max_content_tokens]
            truncated_count += 1
        truncated_texts.append(
            tokenizer.decode(
                token_ids,
                skip_special_tokens=cfg.bertscore_skip_special_tokens,
                clean_up_tokenization_spaces=cfg.bertscore_clean_up_tokenization_spaces,
            )
        )

    return truncated_texts, truncated_count, longest_original_length


def compute_average_bertscore(
    predictions,
    references,
    model_type=None,
    num_layers=None,
    lang="zh",
    batch_size=8,
    device=None,
):
    if bert_score_score is None:
        raise ImportError("bert_score is not installed.")
    if not predictions or not references:
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
        }

    tokenizer = None
    max_content_tokens = None
    if model_type:
        tokenizer, max_content_tokens = load_bertscore_tokenizer_and_limit(model_type)
        predictions, truncated_predictions, max_prediction_length = truncate_texts_for_bertscore(
            predictions,
            tokenizer,
            max_content_tokens,
        )
        references, truncated_references, max_reference_length = truncate_texts_for_bertscore(
            references,
            tokenizer,
            max_content_tokens,
        )
        if max_content_tokens is not None:
            print(
                "BERTScore text truncation: "
                f"limit={max_content_tokens} tokens, "
                f"truncated_predictions={truncated_predictions}, "
                f"truncated_references={truncated_references}, "
                f"max_prediction_tokens={max_prediction_length}, "
                f"max_reference_tokens={max_reference_length}"
            )

    score_kwargs = {
        "cands": predictions,
        "refs": references,
        "lang": lang,
        "batch_size": batch_size,
        "verbose": False,
    }
    if model_type:
        score_kwargs["model_type"] = model_type
    if num_layers is not None:
        score_kwargs["num_layers"] = num_layers
    if device:
        score_kwargs["device"] = device

    precision, recall, f1 = bert_score_score(**score_kwargs)
    return {
        "bertscore_precision": float(precision.mean().item() * 100),
        "bertscore_recall": float(recall.mean().item() * 100),
        "bertscore_f1": float(f1.mean().item() * 100),
    }


def split_text_for_segmented_bertscore(text, tokenizer, chunk_tokens):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [
        (
            tokenizer.decode(
                token_ids[start:start + chunk_tokens],
                skip_special_tokens=cfg.bertscore_skip_special_tokens,
                clean_up_tokenization_spaces=cfg.bertscore_clean_up_tokenization_spaces,
            ),
            len(token_ids[start:start + chunk_tokens]),
        )
        for start in range(0, len(token_ids), chunk_tokens)
    ]


def compute_segmented_bertscore(
    predictions,
    references,
    model_type,
    num_layers,
    lang,
    batch_size,
    chunk_tokens,
    device,
):
    """Compute full-document semantic coverage from pairwise BERTScore chunk matches."""
    tokenizer, max_content_tokens = load_bertscore_tokenizer_and_limit(model_type)
    if tokenizer is None:
        raise ValueError("A local BERTScore tokenizer is required for segmented BERTScore.")
    if max_content_tokens is not None and chunk_tokens > max_content_tokens:
        raise ValueError(
            f"segmented_bertscore_chunk_tokens={chunk_tokens} exceeds {max_content_tokens}."
        )

    pair_candidates = []
    pair_references = []
    document_specs = []
    for prediction, reference in zip(predictions, references):
        prediction_chunks = split_text_for_segmented_bertscore(
            prediction, tokenizer, chunk_tokens
        )
        reference_chunks = split_text_for_segmented_bertscore(
            reference, tokenizer, chunk_tokens
        )
        if not prediction_chunks or not reference_chunks:
            document_specs.append(None)
            continue

        start_index = len(pair_candidates)
        for prediction_chunk, _ in prediction_chunks:
            for reference_chunk, _ in reference_chunks:
                pair_candidates.append(prediction_chunk)
                pair_references.append(reference_chunk)
        document_specs.append({
            "start": start_index,
            "prediction_weights": [weight for _, weight in prediction_chunks],
            "reference_weights": [weight for _, weight in reference_chunks],
        })

    pair_scores = []
    if pair_candidates:
        score_kwargs = {
            "cands": pair_candidates,
            "refs": pair_references,
            "lang": lang,
            "batch_size": batch_size,
            "verbose": False,
            "model_type": model_type,
            "device": device,
        }
        if num_layers is not None:
            score_kwargs["num_layers"] = num_layers
        _, _, f1 = bert_score_score(**score_kwargs)
        pair_scores = [max(0.0, min(1.0, value)) for value in f1.tolist()]

    document_precision = []
    document_recall = []
    document_f1 = []
    for prediction, reference, spec in zip(predictions, references, document_specs):
        if spec is None:
            score = 1.0 if not prediction.strip() and not reference.strip() else 0.0
            document_precision.append(score)
            document_recall.append(score)
            document_f1.append(score)
            continue

        prediction_weights = spec["prediction_weights"]
        reference_weights = spec["reference_weights"]
        prediction_count = len(prediction_weights)
        reference_count = len(reference_weights)
        start = spec["start"]
        matrix = [
            pair_scores[
                start + prediction_index * reference_count:
                start + (prediction_index + 1) * reference_count
            ]
            for prediction_index in range(prediction_count)
        ]
        precision = sum(
            max(matrix[index]) * prediction_weights[index]
            for index in range(prediction_count)
        ) / sum(prediction_weights)
        recall = sum(
            max(matrix[prediction_index][reference_index] for prediction_index in range(prediction_count))
            * reference_weights[reference_index]
            for reference_index in range(reference_count)
        ) / sum(reference_weights)
        f1_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        document_precision.append(precision)
        document_recall.append(recall)
        document_f1.append(f1_score)

    denominator = max(len(predictions), 1)
    return {
        "bertscore_segmented_precision": sum(document_precision) / denominator * 100,
        "bertscore_segmented_recall": sum(document_recall) / denominator * 100,
        "bertscore_segmented_f1": sum(document_f1) / denominator * 100,
    }


class RougeSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        eval_generation_dataset=None,
        final_generation_dataset=None,
        generation_max_new_tokens=1024,
        generation_progress_every=10,
        generation_do_sample=False,
        generation_add_generation_prompt=True,
        generation_skip_special_tokens=True,
        enable_bertscore=False,
        bertscore_model_type=None,
        bertscore_num_layers=None,
        bertscore_lang="zh",
        bertscore_batch_size=8,
        enable_segmented_bertscore=False,
        segmented_bertscore_chunk_tokens=450,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_generation_dataset = eval_generation_dataset
        self.final_generation_dataset = final_generation_dataset
        self.generation_max_new_tokens = generation_max_new_tokens
        self.generation_progress_every = generation_progress_every
        self.generation_do_sample = generation_do_sample
        self.generation_add_generation_prompt = generation_add_generation_prompt
        self.generation_skip_special_tokens = generation_skip_special_tokens
        self.enable_bertscore = enable_bertscore
        self.bertscore_model_type = bertscore_model_type
        self.bertscore_num_layers = bertscore_num_layers
        self.bertscore_lang = bertscore_lang
        self.bertscore_batch_size = bertscore_batch_size
        self.enable_segmented_bertscore = enable_segmented_bertscore
        self.segmented_bertscore_chunk_tokens = segmented_bertscore_chunk_tokens
        self.eval_tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        self._run_final_generation_eval = False
        self._last_generation_eval_was_final = False
        self._last_generation_records = []
        self._bertscore_warning_shown = False

    def request_final_generation_eval(self):
        self._run_final_generation_eval = True

    def save_generation_predictions(self, output_path, run_metadata=None):
        if not self._last_generation_eval_was_final:
            raise RuntimeError("Refuse to save predictions before the final generation evaluation.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temporary_path = f"{output_path}.tmp"
        metadata = dict(run_metadata or {})
        with open(temporary_path, "w", encoding="utf-8") as output_file:
            for record in self._last_generation_records:
                payload = {**metadata, **record}
                output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        os.replace(temporary_path, output_path)
        print(
            f"Saved {len(self._last_generation_records)} final predictions to {output_path}"
        )

    def _warn_bertscore_once(self, message):
        if self._bertscore_warning_shown:
            return
        print(message)
        self._bertscore_warning_shown = True

    def _collect_generation_predictions(self):
        if self.eval_generation_dataset is None or self.eval_tokenizer is None:
            return None, None

        is_final_eval = self._run_final_generation_eval
        self._run_final_generation_eval = False
        self._last_generation_eval_was_final = is_final_eval
        generation_dataset = (
            self.final_generation_dataset
            if is_final_eval and self.final_generation_dataset is not None
            else self.eval_generation_dataset
        )
        total_examples = len(generation_dataset)
        if total_examples == 0:
            return [], []

        predictions = []
        references = []
        generation_records = []
        model_device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        eval_scope = "final" if is_final_eval else "checkpoint-selection"
        print(
            f"Start generating predictions for {eval_scope} metrics on "
            f"{total_examples} eval samples..."
        )

        try:
            for index, example in enumerate(generation_dataset):
                if index >= total_examples:
                    break

                conversation = example["conversations"]
                prompt_messages = conversation[:-1]
                reference_text = str(
                    example.get("source_reference", conversation[-1]["content"])
                ).strip()
                prompt_text = self.eval_tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=self.generation_add_generation_prompt,
                )
                inputs = self.eval_tokenizer(prompt_text, return_tensors="pt")
                inputs = {key: value.to(model_device) for key, value in inputs.items()}

                pad_token_id = self.eval_tokenizer.pad_token_id
                eos_token_id = getattr(self.eval_tokenizer, "eos_token_id", None)
                if pad_token_id is None:
                    pad_token_id = eos_token_id

                generation_kwargs = {
                    "max_new_tokens": self.generation_max_new_tokens,
                    "do_sample": self.generation_do_sample,
                    "pad_token_id": pad_token_id,
                }
                if eos_token_id is not None:
                    generation_kwargs["eos_token_id"] = eos_token_id

                with torch.inference_mode():
                    generated = self.model.generate(**inputs, **generation_kwargs)

                generated_tokens = generated[0, inputs["input_ids"].shape[1]:]
                prediction_text = self.eval_tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=self.generation_skip_special_tokens,
                ).strip()

                predictions.append(prediction_text)
                references.append(reference_text)
                source_system = str(example.get("source_system", "") or "").strip()
                source_instruction = str(
                    example.get("source_instruction", "") or ""
                ).strip()
                source_input = str(example.get("source_input", "") or "").strip()
                question = (
                    f"{source_instruction}\n{source_input}"
                    if source_input
                    else source_instruction
                )
                generation_records.append({
                    "sample_id": index,
                    "system": source_system,
                    "instruction": source_instruction,
                    "input": source_input,
                    "question": question,
                    "reference": reference_text,
                    "prediction": prediction_text,
                    "generation_max_new_tokens": self.generation_max_new_tokens,
                })

                current = index + 1
                if current % self.generation_progress_every == 0 or current == total_examples:
                    print(f"Generation metric progress: {current}/{total_examples}")
        finally:
            if was_training:
                self.model.train()

        self._last_generation_records = generation_records
        return predictions, references

    def _compute_generation_metrics(self, metric_key_prefix="eval"):
        predictions, references = self._collect_generation_predictions()
        if predictions is None or references is None:
            return {}

        metrics = {}
        rouge_scores = compute_average_rouge(predictions, references)
        metrics.update({
            f"{metric_key_prefix}_rouge1": rouge_scores["rouge1"],
            f"{metric_key_prefix}_rouge2": rouge_scores["rouge2"],
            f"{metric_key_prefix}_rougeL": rouge_scores["rougeL"],
            f"{metric_key_prefix}_bleu4": compute_corpus_bleu4(predictions, references),
        })

        if self.enable_bertscore:
            if bert_score_score is None:
                message = (
                    "BERTScore is enabled but the 'bert_score' package is not installed."
                )
                if "bertscore" in self.args.metric_for_best_model:
                    raise RuntimeError(message + " It is required for best-model selection.")
                self._warn_bertscore_once(message + " Skip BERTScore evaluation.")
            else:
                try:
                    print(
                        "Start BERTScore evaluation on "
                        f"{len(predictions)} samples with model={self.bertscore_model_type}, "
                        f"num_layers={self.bertscore_num_layers}, batch_size={self.bertscore_batch_size}."
                    )
                    bertscore_scores = compute_average_bertscore(
                        predictions,
                        references,
                        model_type=self.bertscore_model_type,
                        num_layers=self.bertscore_num_layers,
                        lang=self.bertscore_lang,
                        batch_size=self.bertscore_batch_size,
                        device=str(next(self.model.parameters()).device),
                    )
                    print(
                        "Finished BERTScore evaluation: "
                        f"P={bertscore_scores['bertscore_precision']:.4f}, "
                        f"R={bertscore_scores['bertscore_recall']:.4f}, "
                        f"F1={bertscore_scores['bertscore_f1']:.4f}"
                    )
                    metrics.update({
                        f"{metric_key_prefix}_bertscore_precision": bertscore_scores["bertscore_precision"],
                        f"{metric_key_prefix}_bertscore_recall": bertscore_scores["bertscore_recall"],
                        f"{metric_key_prefix}_bertscore_f1": bertscore_scores["bertscore_f1"],
                    })
                    if self.enable_segmented_bertscore:
                        eval_scope = (
                            "final"
                            if self._last_generation_eval_was_final
                            else "checkpoint-selection"
                        )
                        print(
                            f"Start {eval_scope} segmented BERTScore evaluation with "
                            f"chunk_tokens={self.segmented_bertscore_chunk_tokens}."
                        )
                        segmented_scores = compute_segmented_bertscore(
                            predictions,
                            references,
                            model_type=self.bertscore_model_type,
                            num_layers=self.bertscore_num_layers,
                            lang=self.bertscore_lang,
                            batch_size=self.bertscore_batch_size,
                            chunk_tokens=self.segmented_bertscore_chunk_tokens,
                            device=str(next(self.model.parameters()).device),
                        )
                        metrics.update({
                            f"{metric_key_prefix}_{key}": value
                            for key, value in segmented_scores.items()
                        })
                except Exception as error:
                    if "bertscore" in self.args.metric_for_best_model:
                        raise RuntimeError(
                            "BERTScore evaluation failed, so best-model selection cannot continue."
                        ) from error
                    self._warn_bertscore_once(
                        f"BERTScore evaluation failed and will be skipped: {error}"
                    )

        return metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        generation_metrics = self._compute_generation_metrics(metric_key_prefix=metric_key_prefix)
        metrics.update(generation_metrics)
        if generation_metrics:
            print(generation_metrics)
            self.log(generation_metrics)
        return metrics


def find_resume_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    return get_last_checkpoint(output_dir)


def has_adapter_weights(checkpoint_dir):
    adapter_weight_files = ("adapter_model.safetensors", "adapter_model.bin")
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return False
    return any(os.path.exists(os.path.join(checkpoint_dir, file_name)) for file_name in adapter_weight_files)


def read_best_adapter_metadata(best_adapter_dir):
    if not best_adapter_dir:
        return {}
    metadata_path = os.path.join(best_adapter_dir, "best_checkpoint_info.json")
    if not os.path.exists(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def persist_adapter_artifacts(output_dir, tokenizer, metadata, model=None, source_checkpoint=None):
    os.makedirs(output_dir, exist_ok=True)

    if source_checkpoint is not None:
        copied_files = []
        for file_name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"):
            source_path = os.path.join(source_checkpoint, file_name)
            if os.path.exists(source_path):
                shutil.copy2(source_path, os.path.join(output_dir, file_name))
                copied_files.append(file_name)
        if not copied_files:
            raise FileNotFoundError(f"No adapter artifacts found in checkpoint: {source_checkpoint}")
    elif model is not None:
        model.save_pretrained(output_dir)
    else:
        raise ValueError("Either model or source_checkpoint must be provided.")

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    metadata_path = os.path.join(output_dir, "best_checkpoint_info.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def find_best_available_checkpoint_from_state(trainer_state_path, trainer_output_dir, metric_name):
    if not trainer_state_path or not os.path.exists(trainer_state_path):
        return None

    with open(trainer_state_path, "r", encoding="utf-8") as file:
        trainer_state = json.load(file)

    best_candidate = None
    for entry in trainer_state.get("log_history", []):
        metric_value = entry.get(metric_name)
        step = entry.get("step")
        if metric_value is None or step is None:
            continue

        checkpoint_dir = os.path.join(trainer_output_dir, f"checkpoint-{step}")
        if not has_adapter_weights(checkpoint_dir):
            continue

        if best_candidate is None or metric_value > best_candidate["metric"]:
            best_candidate = {
                "metric": metric_value,
                "step": step,
                "checkpoint_dir": checkpoint_dir,
            }

    return best_candidate


def reconcile_resume_best_checkpoint(
    resume_checkpoint,
    trainer_output_dir,
    best_adapter_dir,
    tokenizer,
    metric_name="eval_rougeL",
):
    if not resume_checkpoint:
        return

    trainer_state_path = os.path.join(resume_checkpoint, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        return

    with open(trainer_state_path, "r", encoding="utf-8") as file:
        trainer_state = json.load(file)

    recorded_best_checkpoint = trainer_state.get("best_model_checkpoint")
    metadata = read_best_adapter_metadata(best_adapter_dir)

    if has_adapter_weights(best_adapter_dir):
        trainer_state["best_model_checkpoint"] = best_adapter_dir
        if metadata.get("best_metric") is not None:
            trainer_state["best_metric"] = metadata["best_metric"]
        if metadata.get("best_global_step") is not None:
            trainer_state["best_global_step"] = metadata["best_global_step"]
        with open(trainer_state_path, "w", encoding="utf-8") as file:
            json.dump(trainer_state, file, ensure_ascii=False, indent=2)
        print(f"Reuse persisted best adapter: {best_adapter_dir}")
        return None

    if has_adapter_weights(recorded_best_checkpoint):
        persist_adapter_artifacts(
            output_dir=best_adapter_dir,
            tokenizer=tokenizer,
            source_checkpoint=recorded_best_checkpoint,
            metadata={
                "best_metric": trainer_state.get("best_metric"),
                "best_global_step": trainer_state.get("best_global_step"),
                "source_checkpoint": recorded_best_checkpoint,
            },
        )
        trainer_state["best_model_checkpoint"] = best_adapter_dir
        with open(trainer_state_path, "w", encoding="utf-8") as file:
            json.dump(trainer_state, file, ensure_ascii=False, indent=2)
        print(f"Bootstrap persisted best adapter from retained checkpoint: {recorded_best_checkpoint}")
        return None

    recovered_checkpoint = find_best_available_checkpoint_from_state(
        trainer_state_path=trainer_state_path,
        trainer_output_dir=trainer_output_dir,
        metric_name=metric_name,
    )
    if recovered_checkpoint is None:
        print("Best checkpoint record is stale and no retained checkpoint can be recovered.")
        return None

    persist_adapter_artifacts(
        output_dir=best_adapter_dir,
        tokenizer=tokenizer,
        source_checkpoint=recovered_checkpoint["checkpoint_dir"],
        metadata={
            "best_metric": recovered_checkpoint["metric"],
            "best_global_step": recovered_checkpoint["step"],
            "source_checkpoint": recovered_checkpoint["checkpoint_dir"],
            "recovered_from": trainer_state_path,
        },
    )
    trainer_state["best_model_checkpoint"] = best_adapter_dir
    trainer_state["best_metric"] = recovered_checkpoint["metric"]
    trainer_state["best_global_step"] = recovered_checkpoint["step"]
    with open(trainer_state_path, "w", encoding="utf-8") as file:
        json.dump(trainer_state, file, ensure_ascii=False, indent=2)
    print(
        f"Recovered best checkpoint from retained adapter: "
        f"{recovered_checkpoint['checkpoint_dir']} -> {best_adapter_dir}"
    )
    return None


def resolve_best_adapter_dir(best_adapter_dir, checkpoint_dir):
    for candidate in (best_adapter_dir, checkpoint_dir):
        if has_adapter_weights(candidate):
            return candidate
    return None


def load_best_adapter_checkpoint(model, checkpoint_dir, best_adapter_dir=None, adapter_name="best_checkpoint"):
    resolved_checkpoint_dir = resolve_best_adapter_dir(best_adapter_dir, checkpoint_dir)
    if not resolved_checkpoint_dir:
        print("No available best adapter found, skip reload.")
        return None

    if not hasattr(model, "load_adapter") or not hasattr(model, "set_adapter"):
        print("Current model does not support load_adapter/set_adapter, skip reloading best adapter.")
        return None

    existing_adapters = set(getattr(model, "peft_config", {}).keys())
    selected_adapter_name = adapter_name
    suffix = 1
    while selected_adapter_name in existing_adapters:
        selected_adapter_name = f"{adapter_name}_{suffix}"
        suffix += 1

    model.load_adapter(resolved_checkpoint_dir, adapter_name=selected_adapter_name, is_trainable=False)
    model.set_adapter(selected_adapter_name)
    if hasattr(model, "set_requires_grad"):
        model.set_requires_grad(selected_adapter_name, requires_grad=False)

    print(f"Loaded best adapter from {resolved_checkpoint_dir} as {selected_adapter_name}.")
    return selected_adapter_name


class PersistBestAdapterCallback(TrainerCallback):
    def __init__(self, best_adapter_dir, tokenizer, metric_name="eval_rougeL"):
        self.best_adapter_dir = best_adapter_dir
        self.tokenizer = tokenizer
        self.metric_name = metric_name
        self.last_saved_step = None
        self.last_saved_metric = None

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None or state.best_model_checkpoint is None or state.best_metric is None:
            return control

        current_checkpoint = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.normpath(state.best_model_checkpoint) != os.path.normpath(current_checkpoint):
            return control

        metric_value = float(state.best_metric)
        if self.last_saved_step == state.global_step and self.last_saved_metric == metric_value:
            return control

        persist_adapter_artifacts(
            output_dir=self.best_adapter_dir,
            tokenizer=self.tokenizer,
            model=model,
            metadata={
                "best_metric": metric_value,
                "best_global_step": state.global_step,
                "source_checkpoint": current_checkpoint,
                "metric_name": self.metric_name,
            },
        )
        self.last_saved_step = state.global_step
        self.last_saved_metric = metric_value
        print(
            f"Persisted best adapter to {self.best_adapter_dir} "
            f"({self.metric_name}={metric_value:.4f}, step={state.global_step})."
        )
        return control


class PlateauEarlyStoppingCallback(TrainerCallback, ExportableState):
    def __init__(self, metric_name="eval_rougeL", patience=5, min_delta=0.0):
        self.metric_name = metric_name
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.bad_eval_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Generation metrics are logged after the base Trainer fires on_evaluate,
        # so early stopping must consume the generation-metric log event.
        if logs is None or self.metric_name not in logs:
            return control

        current_metric = float(logs[self.metric_name])
        if self.best_metric is None or current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.bad_eval_count = 0
            return control

        self.bad_eval_count += 1
        print(
            f"No {self.metric_name} improvement for {self.bad_eval_count}/"
            f"{self.patience} evaluations. Current = {current_metric:.4f}, "
            f"Best = {self.best_metric:.4f}"
        )
        if self.bad_eval_count >= self.patience:
            control.should_training_stop = True
            print(
                f"Early stopping triggered because {self.metric_name} did not improve "
                f"for {self.patience} consecutive evaluations."
            )
        return control

    def state(self):
        return {
            "args": {
                "metric_name": self.metric_name,
                "patience": self.patience,
                "min_delta": self.min_delta,
            },
            "attributes": {
                "best_metric": self.best_metric,
                "bad_eval_count": self.bad_eval_count,
            },
        }


def load_training_adapter_into_default(model, adapter_dir):
    if not has_adapter_weights(adapter_dir):
        print(f"Bootstrap adapter not found, skip warm start: {adapter_dir}")
        return False

    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(safetensors_path):
        if safe_load_file is None:
            raise ImportError("safetensors is required to load adapter_model.safetensors")
        adapter_state_dict = safe_load_file(safetensors_path)
    elif os.path.exists(bin_path):
        adapter_state_dict = torch.load(bin_path, map_location="cpu")
    else:
        print(f"No adapter weight file found under {adapter_dir}")
        return False

    set_peft_model_state_dict(model, adapter_state_dict, adapter_name="default")
    print(f"Loaded bootstrap adapter weights into trainable default adapter: {adapter_dir}")
    return True


def sha256_file(file_path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(file_path):
    resolved_path = os.path.realpath(file_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Required resume fingerprint file is missing: {resolved_path}")
    return {
        "path": resolved_path,
        "size": os.path.getsize(resolved_path),
        "sha256": sha256_file(resolved_path),
    }


def hash_json(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_resume_manifest(config_dict):
    adapter_weight_path = next(
        (
            os.path.join(cfg.bootstrap_adapter_dir, file_name)
            for file_name in ("adapter_model.safetensors", "adapter_model.bin")
            if os.path.isfile(os.path.join(cfg.bootstrap_adapter_dir, file_name))
        ),
        None,
    )
    if cfg.bootstrap_enabled and adapter_weight_path is None:
        raise FileNotFoundError(
            f"Bootstrap is enabled but no adapter weights were found: {cfg.bootstrap_adapter_dir}"
        )

    lora_structure = {
        "r": cfg.lora_r,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "bias": cfg.lora_bias,
        "target_modules": sorted(cfg.lora_target_modules),
        "use_rslora": cfg.use_rslora,
        "loftq_config": cfg.loftq_config,
    }
    bootstrap = {
        "enabled": cfg.bootstrap_enabled,
        "path": os.path.realpath(cfg.bootstrap_adapter_dir),
    }
    if cfg.bootstrap_enabled:
        bootstrap["adapter_config"] = fingerprint_file(
            os.path.join(cfg.bootstrap_adapter_dir, "adapter_config.json")
        )
        bootstrap["adapter_weights"] = fingerprint_file(adapter_weight_path)

    return {
        "schema_version": 2,
        "training_framework": {
            "version": TRAINING_FRAMEWORK_VERSION,
            "bootstrap_on": fingerprint_file(os.path.abspath(__file__)),
        },
        "config_sha256": hash_json(config_dict),
        "datasets": {
            "train": fingerprint_file(cfg.train_dataset_path),
            "eval": fingerprint_file(cfg.eval_dataset_path),
        },
        "model": {
            "path": os.path.realpath(cfg.model_path),
            "config": fingerprint_file(os.path.join(cfg.model_path, "config.json")),
        },
        "bootstrap_adapter": bootstrap,
        "lora_structure": lora_structure,
    }


def validate_resume_manifest(config_snapshot_path, current_manifest):
    if not os.path.isfile(config_snapshot_path):
        raise RuntimeError(
            "Refuse to resume: training_config.json is missing, so compatibility cannot be verified."
        )
    with open(config_snapshot_path, "r", encoding="utf-8") as file:
        snapshot = json.load(file)
    previous_manifest = snapshot.get("resume_manifest")
    if previous_manifest is None:
        raise RuntimeError(
            "Refuse to resume legacy checkpoint without a resume manifest. "
            "Use a new experiment_name or explicitly migrate the checkpoint."
        )

    changed_sections = [
        section
        for section in (
            "schema_version", "training_framework", "config_sha256", "datasets",
            "model", "bootstrap_adapter", "lora_structure"
        )
        if previous_manifest.get(section) != current_manifest.get(section)
    ]
    if changed_sections:
        raise RuntimeError(
            "Refuse to resume because the experiment is incompatible with its checkpoint. "
            f"Changed sections: {', '.join(changed_sections)}. "
            "Use a new experiment_name for the changed experiment."
        )
    print("Resume compatibility check passed: config, datasets, model, bootstrap adapter, and LoRA match.")


# ============== 1.从独立配置组装本次实验 ================================
local_model_path = cfg.model_path
bootstrap_adapter_dir = cfg.bootstrap_adapter_dir
experiment_name = cfg.experiment_name
adapter_output_dir = os.path.join(cfg.output_root, experiment_name)
best_adapter_dir = f"{adapter_output_dir}_best"
trainer_output_dir = f"{adapter_output_dir}_checkpoints"
train_dataset_path = cfg.train_dataset_path
eval_dataset_path = cfg.eval_dataset_path

os.makedirs(trainer_output_dir, exist_ok=True)
config_snapshot_path = os.path.join(trainer_output_dir, "training_config.json")
existing_checkpoint = find_resume_checkpoint(trainer_output_dir)
if existing_checkpoint and not cfg.resume_enabled:
    raise RuntimeError(
        "Checkpoints already exist but resume_enabled=False. "
        "Use a new experiment_name instead of mixing a fresh run with an existing run."
    )
resume_checkpoint = existing_checkpoint if cfg.resume_enabled else None
resume_manifest = build_resume_manifest(selected_config_dict)
if resume_checkpoint:
    validate_resume_manifest(config_snapshot_path, resume_manifest)
else:
    with open(config_snapshot_path, "w", encoding="utf-8") as config_file:
        json.dump(
            {
                "config_file": selected_config_path,
                "parameters": selected_config_dict,
                "resume_manifest": resume_manifest,
            },
            config_file,
            ensure_ascii=False,
            indent=2,
        )

print(f"Use training config: {selected_config_path}")
print(f"Experiment name: {experiment_name}")
print(f"Config snapshot: {config_snapshot_path}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=local_model_path,
    max_seq_length=cfg.max_seq_length,
    dtype=cfg.model_dtype,
    load_in_4bit=cfg.load_in_4bit,
    load_in_8bit=cfg.load_in_8bit,
    full_finetuning=cfg.full_finetuning,
    local_files_only=cfg.local_files_only,
)

# 2.LoRA结构完全由配置决定
model = FastLanguageModel.get_peft_model(
    model,
    r=cfg.lora_r,
    target_modules=cfg.lora_target_modules,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    bias=cfg.lora_bias,
    use_gradient_checkpointing=cfg.use_gradient_checkpointing,
    random_state=cfg.lora_random_state,
    use_rslora=cfg.use_rslora,
    loftq_config=cfg.loftq_config,
)


if resume_checkpoint:
    reconcile_resume_best_checkpoint(
        resume_checkpoint=resume_checkpoint,
        trainer_output_dir=trainer_output_dir,
        best_adapter_dir=best_adapter_dir,
        tokenizer=tokenizer,
        metric_name=cfg.metric_for_best_model,
    )
    print(f"Resume training from checkpoint: {resume_checkpoint}")
else:
    if cfg.bootstrap_enabled:
        print("No checkpoint found for the new experiment, start from bootstrap adapter.")
        load_training_adapter_into_default(model, bootstrap_adapter_dir)
    else:
        print("No checkpoint found and bootstrap is disabled; start from the base model.")


# ===================== 2.数据加载与格式转换 =========================
"""alpaca数据集处理"""


def build_prompt_completion(example):
    prompts = []
    completions = []
    conversations = []
    source_systems = []
    source_instructions = []
    source_inputs = []
    source_references = []
    for system_text, instruction, input_text, output in zip(
        example[cfg.system_column],
        example[cfg.instruction_column],
        example[cfg.input_column],
        example[cfg.output_column],
    ):
        user_query = f"{instruction}\n{input_text}" if input_text else instruction
        prompt_messages = []
        if system_text and system_text.strip():
            prompt_messages.append({"role": "system", "content": system_text.strip()})
        prompt_messages.append({"role": "user", "content": user_query})
        completion_messages = [
            {"role": "assistant", "content": output},
        ]
        prompts.append(prompt_messages)
        completions.append(completion_messages)
        conversations.append(prompt_messages + completion_messages)
        source_systems.append(system_text)
        source_instructions.append(instruction)
        source_inputs.append(input_text)
        source_references.append(output)
    return {
        "prompt": prompts,
        "completion": completions,
        "conversations": conversations,
        "source_system": source_systems,
        "source_instruction": source_instructions,
        "source_input": source_inputs,
        "source_reference": source_references,
    }


train_raw_dataset = load_dataset(
    "json",
    data_files=train_dataset_path,
    split=cfg.dataset_split,
)
eval_raw_dataset = load_dataset(
    "json",
    data_files=eval_dataset_path,
    split=cfg.dataset_split,
)
print(f"Use train dataset: {train_dataset_path}")
print(f"Use eval dataset: {eval_dataset_path}")
print(train_raw_dataset[0])

train_structured_dataset = train_raw_dataset.map(
    build_prompt_completion,
    batched=cfg.dataset_batched,
    remove_columns=train_raw_dataset.column_names,
)
eval_structured_dataset = eval_raw_dataset.map(
    build_prompt_completion,
    batched=cfg.dataset_batched,
    remove_columns=eval_raw_dataset.column_names,
)

def tokenize_completion_only(example):
    """Tokenize one chat and mask prompt tokens so loss uses only the answer."""
    prompt_messages = example["prompt"]
    completion_messages = example["completion"]
    conversation = prompt_messages + completion_messages

    prompt_input_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=cfg.training_max_length,
    )
    prompt_length = min(len(prompt_input_ids), len(input_ids))
    if input_ids[:prompt_length] != prompt_input_ids[:prompt_length]:
        raise ValueError(
            "Prompt tokens are not a prefix of the full conversation. "
            "Cannot build a reliable completion-only loss mask."
        )

    labels = [-100] * prompt_length + input_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        raise ValueError(
            "A training sample has no completion tokens after truncation. "
            "Increase training_max_length or shorten its prompt."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


train_dataset = train_structured_dataset.map(
    tokenize_completion_only,
    batched=False,
    remove_columns=train_structured_dataset.column_names,
    desc="Tokenizing train conversations with completion-only labels",
)
eval_dataset = eval_structured_dataset.map(
    tokenize_completion_only,
    batched=False,
    remove_columns=eval_structured_dataset.column_names,
    desc="Tokenizing eval conversations with completion-only labels",
)
selection_size = min(cfg.selection_eval_max_samples, len(eval_structured_dataset))
selection_eval_structured_dataset = eval_structured_dataset.shuffle(
    seed=cfg.selection_eval_seed
).select(range(selection_size))
selection_eval_conversation_dataset = selection_eval_structured_dataset.remove_columns(
    ["prompt", "completion"]
)
final_eval_conversation_dataset = eval_structured_dataset.remove_columns(
    ["prompt", "completion"]
)

print(train_dataset[0])
print(f"train samples = {len(train_dataset)}")
print(f"eval samples = {len(eval_dataset)}")
first_train_labels = train_dataset[0]["labels"]
first_completion_tokens = sum(label != -100 for label in first_train_labels)
print(
    "completion-only tokenization check: "
    f"sequence_tokens={len(first_train_labels)}, "
    f"prompt_masked_tokens={len(first_train_labels) - first_completion_tokens}, "
    f"completion_tokens={first_completion_tokens}"
)
print(
    "checkpoint-selection samples = "
    f"{selection_size} (fixed shuffle seed={cfg.selection_eval_seed})"
)
if len(train_dataset) != cfg.expected_train_samples:
    raise ValueError(
        f"Train sample count mismatch: expected {cfg.expected_train_samples}, got {len(train_dataset)}."
    )
if len(eval_dataset) != cfg.expected_eval_samples:
    raise ValueError(
        f"Eval sample count mismatch: expected {cfg.expected_eval_samples}, got {len(eval_dataset)}."
    )
effective_batch_size = (
    cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
)
micro_batches_per_epoch = math.ceil(
    len(train_dataset) / cfg.per_device_train_batch_size
)
optimizer_steps_per_epoch = math.ceil(
    micro_batches_per_epoch / cfg.gradient_accumulation_steps
)
expected_optimizer_steps = math.ceil(
    optimizer_steps_per_epoch * cfg.num_train_epochs
)
if effective_batch_size != cfg.expected_effective_batch_size:
    raise ValueError(
        "Effective batch size mismatch: "
        f"expected {cfg.expected_effective_batch_size}, got {effective_batch_size}."
    )
if expected_optimizer_steps != cfg.expected_optimizer_steps:
    raise ValueError(
        "Optimizer-step estimate mismatch: "
        f"expected {cfg.expected_optimizer_steps}, got {expected_optimizer_steps}."
    )
print(f"effective batch size (single GPU) = {effective_batch_size}")
print(f"expected optimizer steps = {expected_optimizer_steps}")


# =========== 3.使用Transformer Reinforcement Learning（trl）库的监督微调SFT训练器 ==============
trainer_callbacks = [
    PersistBestAdapterCallback(
        best_adapter_dir=best_adapter_dir,
        tokenizer=tokenizer,
        metric_name=cfg.metric_for_best_model,
    ),
]
if cfg.early_stopping_enabled:
    trainer_callbacks.append(
        PlateauEarlyStoppingCallback(
            metric_name=cfg.metric_for_best_model,
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
        )
    )
else:
    print("Early stopping is disabled for this experiment; training will run to the epoch limit.")

trainer = RougeSFTTrainer(
    model=model,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    eval_generation_dataset=selection_eval_conversation_dataset,
    final_generation_dataset=final_eval_conversation_dataset,
    generation_max_new_tokens=cfg.generation_max_new_tokens,
    generation_do_sample=cfg.generation_do_sample,
    generation_add_generation_prompt=cfg.eval_add_generation_prompt,
    generation_skip_special_tokens=cfg.generation_skip_special_tokens,
    generation_progress_every=cfg.generation_progress_every,
    enable_bertscore=cfg.enable_bertscore,
    bertscore_model_type=cfg.bertscore_model_type,
    bertscore_num_layers=cfg.bertscore_num_layers,
    bertscore_lang=cfg.bertscore_lang,
    bertscore_batch_size=cfg.bertscore_batch_size,
    enable_segmented_bertscore=cfg.enable_segmented_bertscore,
    segmented_bertscore_chunk_tokens=cfg.segmented_bertscore_chunk_tokens,
    args=SFTConfig(
        output_dir=trainer_output_dir,
        max_length=cfg.training_max_length,
        packing=cfg.packing,
        completion_only_loss=cfg.completion_only_loss,
        dataset_kwargs={"skip_prepare_dataset": True},
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        report_to=cfg.report_to,
        restore_callback_states_from_checkpoint=True,
    ),
    callbacks=trainer_callbacks,
)


trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
selected_best_checkpoint = resolve_best_adapter_dir(
    best_adapter_dir,
    trainer.state.best_model_checkpoint,
)
selected_best_metric = trainer.state.best_metric
best_adapter_name = load_best_adapter_checkpoint(
    trainer.model,
    checkpoint_dir=trainer.state.best_model_checkpoint,
    best_adapter_dir=best_adapter_dir,
)

if best_adapter_name is None:
    # This is only a fallback for runs that ended before the first scheduled evaluation.
    selection_metrics = trainer.evaluate()
    selected_best_metric = selection_metrics.get(cfg.metric_for_best_model)
    persist_adapter_artifacts(
        output_dir=best_adapter_dir,
        tokenizer=tokenizer,
        model=trainer.model,
        metadata={
            "best_metric": selected_best_metric,
            "best_global_step": trainer.state.global_step,
            "source_checkpoint": f"final-model-step-{trainer.state.global_step}",
            "metric_name": cfg.metric_for_best_model,
        },
    )
    selected_best_checkpoint = best_adapter_dir

# The full validation set is used only to report the selected model, never to replace it.
trainer.request_final_generation_eval()
eval_metrics = trainer.evaluate()
predictions_output_path = os.path.join(adapter_output_dir, "predictions.jsonl")
trainer.save_generation_predictions(
    predictions_output_path,
    run_metadata={
        "experiment_name": experiment_name,
        "config_file": selected_config_path,
        "best_checkpoint": selected_best_checkpoint,
        "selection_metric_name": cfg.metric_for_best_model,
        "selection_metric_value": (
            float(selected_best_metric) if selected_best_metric is not None else None
        ),
    },
)

print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/3600, 2)} h used for training.")
print(f"best checkpoint = {selected_best_checkpoint}")
print(f"best checkpoint-selection {cfg.metric_for_best_model} = {selected_best_metric}")
if "eval_loss" in eval_metrics:
    print(f"eval_loss = {eval_metrics['eval_loss']:.6f}")
    try:
        print(f"eval_perplexity = {math.exp(eval_metrics['eval_loss']):.6f}")
    except OverflowError:
        print("eval_perplexity = inf")
print(f"eval_rouge1 = {eval_metrics['eval_rouge1']:.4f}")
print(f"eval_rouge2 = {eval_metrics['eval_rouge2']:.4f}")
print(f"eval_rougeL = {eval_metrics['eval_rougeL']:.4f}")
print(f"eval_bleu4 = {eval_metrics['eval_bleu4']:.4f}")
if "eval_bertscore_precision" in eval_metrics:
    print(f"eval_bertscore_precision = {eval_metrics['eval_bertscore_precision']:.4f}")
if "eval_bertscore_recall" in eval_metrics:
    print(f"eval_bertscore_recall = {eval_metrics['eval_bertscore_recall']:.4f}")
if "eval_bertscore_f1" in eval_metrics:
    print(f"eval_bertscore_f1 = {eval_metrics['eval_bertscore_f1']:.4f}")
if "eval_bertscore_segmented_precision" in eval_metrics:
    print(
        "eval_bertscore_segmented_precision = "
        f"{eval_metrics['eval_bertscore_segmented_precision']:.4f}"
    )
if "eval_bertscore_segmented_recall" in eval_metrics:
    print(
        "eval_bertscore_segmented_recall = "
        f"{eval_metrics['eval_bertscore_segmented_recall']:.4f}"
    )
if "eval_bertscore_segmented_f1" in eval_metrics:
    print(
        "eval_bertscore_segmented_f1 = "
        f"{eval_metrics['eval_bertscore_segmented_f1']:.4f}"
    )

# ==================== 4.保存训练结果 ====================================
os.makedirs(adapter_output_dir, exist_ok=True)
trainer.model.save_pretrained(
    adapter_output_dir,
    selected_adapters=[best_adapter_name] if best_adapter_name else None,
)

# 同时保存 tokenizer，便于后续推理加载
tokenizer.save_pretrained(
    adapter_output_dir
)
