"""Shared inference implementation for all baseline models."""

import argparse
import json
import math
import os
import types
from collections import Counter
from functools import lru_cache

"""Unsloth must stay before transformers for local-model inference.

API providers only reuse the evaluation functions and set
``INFER_METRICS_ONLY=1`` so importing this module does not require a GPU.
"""
if os.environ.get("INFER_METRICS_ONLY") == "1":
    FastLanguageModel = None
else:
    from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

# Set by main() from the selected model's parameter file.
cfg = None

try:
    from bert_score import score as bert_score_score
except ImportError:
    bert_score_score = None


def parse_args(config):
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--model_path", default=config.model_path, help="Local base model path.")
    parser.add_argument("--eval_path", default=cfg.eval_dataset_path, help="Evaluation dataset path.")
    parser.add_argument("--output_path", default=config.output_path, help="Where to save predictions and metrics.")
    parser.add_argument("--max_samples", type=int, default=cfg.max_samples,
                        help="Limit eval samples for a quick check. Default uses all samples.")
    parser.add_argument("--max_seq_length", type=int, default=cfg.max_seq_length)
    parser.add_argument("--max_new_tokens", type=int, default=cfg.generation_max_new_tokens)
    parser.add_argument("--progress_every", type=int, default=cfg.progress_every)
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=cfg.do_sample,
    )
    parser.add_argument("--temperature", type=float, default=cfg.temperature)
    parser.add_argument("--top_p", type=float, default=cfg.top_p)
    parser.add_argument("--top_k", type=int, default=cfg.top_k)
    parser.add_argument("--bertscore_model_type", default=cfg.bertscore_model_type,
                        help="Local Chinese encoder model path for BERTScore.")
    parser.add_argument("--bertscore_num_layers", type=int, default=cfg.bertscore_num_layers)
    parser.add_argument("--bertscore_lang", default=cfg.bertscore_lang)
    parser.add_argument("--bertscore_batch_size", type=int, default=cfg.bertscore_batch_size)
    parser.add_argument("--no_4bit", action="store_true",
                        default=not cfg.load_in_4bit,
                        help="Disable 4bit loading. Default uses the config setting.")
    args = parser.parse_args()
    args.model_family = config.model_family
    return args

def tokenize_for_rouge(text):
    """Use uniform character-level tokenization and ignore all whitespace."""
    return [char for char in str(text) if not char.isspace()]


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


def compute_corpus_bleu(predictions, references, max_order=4, smooth_value=1.0):
    """Compute single-reference corpus BLEU using clipped n-gram precision."""
    if not predictions or not references:
        return 0.0
    if max_order < 1:
        raise ValueError("BLEU max_order must be at least 1.")
    if smooth_value < 0:
        raise ValueError("BLEU smooth_value cannot be negative.")

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
    for matches, possible in zip(matches_by_order, possible_matches_by_order):
        if possible == 0:
            continue
        if smooth_value > 0:
            precisions.append((matches + smooth_value) / (possible + smooth_value))
        else:
            precisions.append(matches / possible)

    if not precisions or any(precision <= 0 for precision in precisions):
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
        local_files_only=cfg.local_files_only,
        use_fast=cfg.tokenizer_use_fast,
    )
    config = AutoConfig.from_pretrained(
        model_type,
        local_files_only=cfg.local_files_only,
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
                skip_special_tokens=cfg.skip_special_tokens,
                clean_up_tokenization_spaces=cfg.clean_up_tokenization_spaces,
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
                skip_special_tokens=cfg.skip_special_tokens,
                clean_up_tokenization_spaces=cfg.clean_up_tokenization_spaces,
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
    """Compute full-document semantic coverage from pairwise chunk matches."""
    if bert_score_score is None:
        raise ImportError("bert_score is not installed.")
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
            max(
                matrix[prediction_index][reference_index]
                for prediction_index in range(prediction_count)
            ) * reference_weights[reference_index]
            for reference_index in range(reference_count)
        ) / sum(reference_weights)
        f1_score = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        document_precision.append(precision)
        document_recall.append(recall)
        document_f1.append(f1_score)

    denominator = max(len(predictions), 1)
    return {
        "bertscore_segmented_precision": sum(document_precision) / denominator * 100,
        "bertscore_segmented_recall": sum(document_recall) / denominator * 100,
        "bertscore_segmented_f1": sum(document_f1) / denominator * 100,
    }


def safe_text(value):
    return "" if value is None else str(value)


def build_conversations(example):
    conversations = []
    for system_text, instruction, input_text, output in zip(
        example[cfg.system_column],
        example[cfg.instruction_column],
        example[cfg.input_column],
        example[cfg.output_column],
    ):
        instruction = safe_text(instruction).strip()
        input_text = safe_text(input_text).strip()
        output = safe_text(output).strip()
        system_text = safe_text(system_text).strip()
        user_query = instruction
        if input_text:
            user_query = f"{instruction}\n{input_text}" if instruction else input_text
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.extend([
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": output},
        ])
        conversations.append(messages)
    return {"conversations": conversations}

def patch_chatglm_generation_config(model):
    config = getattr(model, "config", None)
    if config is None:
        return
    if not hasattr(config, "num_hidden_layers") and hasattr(config, "num_layers"):
        config.num_hidden_layers = config.num_layers
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = cfg.chatglm_use_cache
    if hasattr(config, "use_cache"):
        config.use_cache = cfg.chatglm_use_cache

    if not hasattr(model, "_extract_past_from_model_output"):
        def _extract_past_from_model_output(self, outputs, *args, **kwargs):
            past = None
            if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                past = outputs.past_key_values
            elif hasattr(outputs, "mems") and outputs.mems is not None:
                past = outputs.mems
            elif hasattr(outputs, "past_buckets_states") and outputs.past_buckets_states is not None:
                past = outputs.past_buckets_states
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 1:
                past = outputs[1]
            return None, past

        model._extract_past_from_model_output = types.MethodType(
            _extract_past_from_model_output,
            model,
        )


def load_base_model(args):
    if FastLanguageModel is None:
        raise RuntimeError(
            "Local model loading is unavailable in INFER_METRICS_ONLY mode."
        )
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_path,
            max_seq_length=args.max_seq_length,
            dtype=cfg.unsloth_dtype,
            load_in_4bit=not args.no_4bit,
            load_in_8bit=cfg.load_in_8bit,
            full_finetuning=False,
            local_files_only=cfg.local_files_only,
        )
        FastLanguageModel.for_inference(model)
        if args.model_family == "chatglm":
            patch_chatglm_generation_config(model)
        return model, tokenizer
    except Exception as error:
        print(f"Unsloth loading failed, fallback to transformers AutoModel: {error}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=cfg.local_files_only,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=cfg.tokenizer_use_fast,
    )
    model_kwargs = {
        "local_files_only": cfg.local_files_only,
        "trust_remote_code": cfg.trust_remote_code,
        "torch_dtype": getattr(torch, cfg.torch_dtype),
        "device_map": cfg.device_map,
    }
    if not args.no_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_compute_dtype),
                bnb_4bit_use_double_quant=cfg.bnb_use_double_quant,
                bnb_4bit_quant_type=cfg.bnb_quant_type,
            )
        except Exception as quant_error:
            print(f"4bit fallback config unavailable, load in fp16 instead: {quant_error}")

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    except Exception as causal_error:
        print(f"AutoModelForCausalLM loading failed, try AutoModel: {causal_error}")
        model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    if args.model_family == "chatglm":
        patch_chatglm_generation_config(model)
    model.eval()
    return model, tokenizer


def print_runtime_info(model, args):
    print(f"model_path = {args.model_path}")
    print(f"eval_path = {args.eval_path}")
    print(f"output_path = {args.output_path}")
    print(f"max_seq_length = {args.max_seq_length}")
    print(f"max_new_tokens = {args.max_new_tokens}")
    print(f"load_in_4bit = {not args.no_4bit}")
    print(f"do_sample = {args.do_sample}")
    print("adapter_path = None (base model only)")
    print("prompt_format = same as train_unsloth_eval300_data7153.py generation eval")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"cuda_available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"current_cuda_device = {current_device}")
        print(f"current_cuda_name = {torch.cuda.get_device_name(current_device)}")
    print(f"model_first_parameter_device = {next(model.parameters()).device}")


def render_prompt(tokenizer, prompt_messages):
    try:
        # Qwen3 defaults to its reasoning mode unless this is explicitly
        # disabled in the chat template. Keep other model families unchanged.
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": cfg.add_generation_prompt,
        }
        enable_thinking = getattr(cfg, "enable_thinking", None)
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(
            prompt_messages,
            **template_kwargs,
        )
    except Exception as error:
        print(f"apply_chat_template failed, fallback to role-labelled prompt: {error}")
        rendered_messages = [
            f"{message['role']}: {message['content']}"
            for message in prompt_messages
        ]
        rendered_messages.append("assistant:")
        return "\n".join(rendered_messages)

def collect_generation_predictions(model, tokenizer, eval_generation_dataset, args):
    total_examples = len(eval_generation_dataset)
    if args.max_samples is not None:
        total_examples = min(total_examples, args.max_samples)
    if total_examples == 0:
        return [], [], []

    predictions = []
    references = []
    records = []
    model_device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    print(f"Start generating predictions for full metrics on {total_examples} eval samples...")

    try:
        for index, example in enumerate(eval_generation_dataset):
            if index >= total_examples:
                break

            conversation = example["conversations"]
            prompt_messages = conversation[:-1]
            reference_text = conversation[-1]["content"].strip()
            prompt_text = render_prompt(tokenizer, prompt_messages)
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

            pad_token_id = tokenizer.pad_token_id
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if pad_token_id is None:
                pad_token_id = eos_token_id

            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "pad_token_id": pad_token_id,
            }
            if args.temperature is not None:
                generation_kwargs["temperature"] = args.temperature
            if args.top_p is not None:
                generation_kwargs["top_p"] = args.top_p
            if args.top_k is not None:
                generation_kwargs["top_k"] = args.top_k
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
            if args.model_family == "chatglm" and not hasattr(
                getattr(model, "config", None), "num_hidden_layers"
            ):
                generation_kwargs["use_cache"] = cfg.chatglm_use_cache

            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)

            generated_tokens = generated[0, inputs["input_ids"].shape[1]:]
            prediction_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=cfg.skip_special_tokens,
            ).strip()

            predictions.append(prediction_text)
            references.append(reference_text)
            records.append({
                "index": index,
                "prompt_messages": prompt_messages,
                "prompt": prompt_text,
                "reference": reference_text,
                "prediction": prediction_text,
            })

            current = index + 1
            if current % args.progress_every == 0 or current == total_examples:
                print(f"Generation metric progress: {current}/{total_examples}")
    finally:
        if was_training:
            model.train()

    return predictions, references, records


def compute_metrics(model, predictions, references, args):
    metrics = {}
    rouge_scores = compute_average_rouge(predictions, references)
    metrics.update({
        "eval_rouge1": rouge_scores["rouge1"],
        "eval_rouge2": rouge_scores["rouge2"],
        "eval_rougeL": rouge_scores["rougeL"],
    })

    if getattr(cfg, "enable_bleu", False):
        bleu_score = compute_corpus_bleu(
            predictions,
            references,
            max_order=cfg.bleu_max_order,
            smooth_value=cfg.bleu_smooth_value,
        )
        metrics["eval_bleu4"] = bleu_score
        print(
            f"eval_bleu4 = {bleu_score:.4f} "
            f"(BLEU-{cfg.bleu_max_order}, smooth={cfg.bleu_smooth_value})"
        )

    if not cfg.enable_bertscore:
        return metrics

    print(
        "Start BERTScore evaluation on "
        f"{len(predictions)} samples with model={args.bertscore_model_type}, "
        f"num_layers={args.bertscore_num_layers}, batch_size={args.bertscore_batch_size}."
    )
    bertscore_scores = compute_average_bertscore(
        predictions,
        references,
        model_type=args.bertscore_model_type,
        num_layers=args.bertscore_num_layers,
        lang=args.bertscore_lang,
        batch_size=args.bertscore_batch_size,
        device=str(next(model.parameters()).device),
    )
    print(
        "Finished BERTScore evaluation: "
        f"P={bertscore_scores['bertscore_precision']:.4f}, "
        f"R={bertscore_scores['bertscore_recall']:.4f}, "
        f"F1={bertscore_scores['bertscore_f1']:.4f}"
    )
    metrics.update({
        "eval_bertscore_precision": bertscore_scores["bertscore_precision"],
        "eval_bertscore_recall": bertscore_scores["bertscore_recall"],
        "eval_bertscore_f1": bertscore_scores["bertscore_f1"],
    })
    if getattr(cfg, "enable_segmented_bertscore", False):
        print(
            "Start segmented BERTScore evaluation with "
            f"chunk_tokens={cfg.segmented_bertscore_chunk_tokens}."
        )
        segmented_scores = compute_segmented_bertscore(
            predictions,
            references,
            model_type=args.bertscore_model_type,
            num_layers=args.bertscore_num_layers,
            lang=args.bertscore_lang,
            batch_size=args.bertscore_batch_size,
            chunk_tokens=cfg.segmented_bertscore_chunk_tokens,
            device=str(next(model.parameters()).device),
        )
        metrics.update({
            f"eval_{key}": value
            for key, value in segmented_scores.items()
        })
    return metrics


def save_results(args, metrics, records):
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "adapter_path": None,
                "eval_path": args.eval_path,
                "num_samples": len(records),
                "max_new_tokens": args.max_new_tokens,
                "decode_strategy": "sampling" if args.do_sample else "greedy",
                "metrics": metrics,
                "records": records,
            },
            f,
            ensure_ascii=cfg.output_ensure_ascii,
            indent=cfg.output_indent,
        )
    print(f"Saved predictions to: {args.output_path}")


def main(config):
    global cfg
    if hasattr(config, "eval_dataset_path"):
        cfg = config
    else:
        # Compatibility for older entries assembled by infer_config.py.
        import infer_config as legacy_config
        cfg = legacy_config

    args = parse_args(config)
    if cfg.enable_bertscore and bert_score_score is None:
        raise ImportError("BERTScore is required for this baseline comparison, but bert_score is not installed.")

    model, tokenizer = load_base_model(args)
    print_runtime_info(model, args)

    eval_raw_dataset = load_dataset(
        "json",
        data_files=args.eval_path,
        split=cfg.dataset_split,
    )
    print(f"Use eval dataset: {args.eval_path}")
    print(eval_raw_dataset[0])

    eval_conversation_dataset = eval_raw_dataset.map(
        build_conversations,
        batched=cfg.dataset_batched,
        remove_columns=eval_raw_dataset.column_names,
    )
    print(f"eval samples = {len(eval_conversation_dataset)}")

    predictions, references, records = collect_generation_predictions(
        model,
        tokenizer,
        eval_conversation_dataset,
        args,
    )
    metrics = compute_metrics(model, predictions, references, args)
    save_results(args, metrics, records)

    for key, value in metrics.items():
        print(f"{key} = {value:.4f}")
