"""Evaluate one fixed LoRA adapter at several generation length limits."""

import argparse
import csv
import importlib.util
import json
import math
import os
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "evaluation_configs" / "generation_lengths.py"


def load_config():
    spec = importlib.util.spec_from_file_location("generation_lengths_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.CONFIG)


def parse_args(config):
    parser = argparse.ArgumentParser(
        description="Generation-length ablation for one fixed trained LoRA adapter."
    )
    parser.add_argument("--adapter-path", default=config.get("adapter_path"))
    parser.add_argument("--base-model-path", default=config["base_model_path"])
    parser.add_argument("--eval-path", default=config["eval_dataset_path"])
    parser.add_argument("--output-dir", default=config["output_dir"])
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=config["max_new_tokens_values"]
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical_path(path):
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def validate_inputs(args):
    if not args.adapter_path:
        raise ValueError("--adapter-path is required; base-model-only ablation is forbidden.")
    args.adapter_path = canonical_path(args.adapter_path)
    args.base_model_path = canonical_path(args.base_model_path)
    args.eval_path = canonical_path(args.eval_path)
    args.output_dir = canonical_path(args.output_dir)
    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_path}")
    adapter_config_path = os.path.join(args.adapter_path, "adapter_config.json")
    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found: {adapter_config_path}")
    if not os.path.isfile(args.eval_path):
        raise FileNotFoundError(f"Evaluation dataset not found: {args.eval_path}")
    if any(length <= 0 for length in args.lengths):
        raise ValueError("All --lengths values must be positive.")
    if len(set(args.lengths)) != len(args.lengths):
        raise ValueError("--lengths contains duplicate values.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")

    with open(adapter_config_path, encoding="utf-8") as file:
        adapter_config = json.load(file)
    recorded_base = adapter_config.get("base_model_name_or_path")
    if recorded_base:
        recorded_base = canonical_path(recorded_base)
        if recorded_base != args.base_model_path:
            raise ValueError(
                "Adapter/base-model mismatch: adapter_config.json records "
                f"{recorded_base}, but --base-model-path is {args.base_model_path}."
            )
    return adapter_config


def tokenize_for_rouge(text):
    """Tokenize Chinese ROUGE/BLEU inputs consistently at character level."""
    return [character for character in text.strip() if not character.isspace()]


def rouge_n_f1(prediction_tokens, reference_tokens, order):
    if len(prediction_tokens) < order or len(reference_tokens) < order:
        return 0.0
    prediction_ngrams = Counter(
        tuple(prediction_tokens[index:index + order])
        for index in range(len(prediction_tokens) - order + 1)
    )
    reference_ngrams = Counter(
        tuple(reference_tokens[index:index + order])
        for index in range(len(reference_tokens) - order + 1)
    )
    overlap = sum((prediction_ngrams & reference_ngrams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(prediction_ngrams.values())
    recall = overlap / sum(reference_ngrams.values())
    return 2 * precision * recall / (precision + recall)


def lcs_length(prediction_tokens, reference_tokens):
    row = [0] * (len(reference_tokens) + 1)
    for prediction_token in prediction_tokens:
        previous = 0
        for index, reference_token in enumerate(reference_tokens, start=1):
            current = row[index]
            if prediction_token == reference_token:
                row[index] = previous + 1
            else:
                row[index] = max(row[index], row[index - 1])
            previous = current
    return row[-1]


def rouge_l_f1(prediction_tokens, reference_tokens):
    if not prediction_tokens or not reference_tokens:
        return 0.0
    length = lcs_length(prediction_tokens, reference_tokens)
    if length == 0:
        return 0.0
    precision = length / len(prediction_tokens)
    recall = length / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge(predictions, references):
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for prediction, reference in zip(predictions, references):
        prediction_tokens = tokenize_for_rouge(prediction)
        reference_tokens = tokenize_for_rouge(reference)
        scores["rouge1"].append(rouge_n_f1(prediction_tokens, reference_tokens, 1))
        scores["rouge2"].append(rouge_n_f1(prediction_tokens, reference_tokens, 2))
        scores["rougeL"].append(rouge_l_f1(prediction_tokens, reference_tokens))
    denominator = max(len(predictions), 1)
    return {key: sum(values) / denominator * 100 for key, values in scores.items()}


def compute_bleu4(predictions, references):
    matches = [0] * 4
    possible = [0] * 4
    prediction_length = 0
    reference_length = 0
    for prediction, reference in zip(predictions, references):
        prediction_tokens = tokenize_for_rouge(prediction)
        reference_tokens = tokenize_for_rouge(reference)
        prediction_length += len(prediction_tokens)
        reference_length += len(reference_tokens)
        for order in range(1, 5):
            prediction_ngrams = Counter(
                tuple(prediction_tokens[index:index + order])
                for index in range(max(len(prediction_tokens) - order + 1, 0))
            )
            reference_ngrams = Counter(
                tuple(reference_tokens[index:index + order])
                for index in range(max(len(reference_tokens) - order + 1, 0))
            )
            matches[order - 1] += sum((prediction_ngrams & reference_ngrams).values())
            possible[order - 1] += sum(prediction_ngrams.values())
    if prediction_length == 0:
        return 0.0
    precisions = []
    for matched, count in zip(matches, possible):
        if count == 0:
            continue
        precision = matched / count
        if precision == 0:
            return 0.0
        precisions.append(precision)
    if not precisions:
        return 0.0
    geometric_mean = math.exp(sum(math.log(value) for value in precisions) / len(precisions))
    brevity_penalty = (
        1.0
        if prediction_length >= reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    return brevity_penalty * geometric_mean * 100


def load_eval_data(path, max_samples):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("Evaluation JSON must contain a top-level list.")
    if max_samples is not None:
        data = data[:max_samples]
    required = {"instruction", "input", "output", "system"}
    for index, item in enumerate(data):
        missing = required - set(item)
        if missing:
            raise ValueError(f"Sample {index} is missing fields: {sorted(missing)}")
    return data


def build_messages(example):
    messages = []
    system_text = str(example.get("system", "")).strip()
    if system_text:
        messages.append({"role": "system", "content": system_text})
    instruction = str(example["instruction"])
    input_text = str(example.get("input", ""))
    user_text = f"{instruction}\n{input_text}" if input_text else instruction
    messages.append({"role": "user", "content": user_text})
    return messages


@lru_cache(maxsize=1)
def load_bertscore_components(model_type):
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_type, local_files_only=True, use_fast=True)
    model_config = AutoConfig.from_pretrained(model_type, local_files_only=True)
    max_positions = getattr(model_config, "max_position_embeddings", None)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if tokenizer_limit is not None and tokenizer_limit > 100000:
        tokenizer_limit = None
    raw_limit = max_positions if max_positions is not None else tokenizer_limit
    content_limit = None
    if raw_limit is not None:
        content_limit = max(raw_limit - tokenizer.num_special_tokens_to_add(pair=False), 1)
    return tokenizer, content_limit


def decode_chunks(text, tokenizer, chunk_tokens):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [
        (
            tokenizer.decode(
                token_ids[start:start + chunk_tokens],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ),
            len(token_ids[start:start + chunk_tokens]),
        )
        for start in range(0, len(token_ids), chunk_tokens)
    ]


def compute_bertscore(predictions, references, config, device):
    from bert_score import score

    tokenizer, content_limit = load_bertscore_components(config["bertscore_model_type"])
    processed = []
    truncation_counts = []
    for texts in (predictions, references):
        values = []
        truncated = 0
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if content_limit is not None and len(token_ids) > content_limit:
                token_ids = token_ids[:content_limit]
                truncated += 1
            values.append(
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        processed.append(values)
        truncation_counts.append(truncated)
    precision, recall, f1 = score(
        cands=processed[0],
        refs=processed[1],
        model_type=config["bertscore_model_type"],
        num_layers=config["bertscore_num_layers"],
        lang=config["bertscore_lang"],
        batch_size=config["bertscore_batch_size"],
        device=device,
        verbose=False,
    )
    return {
        "bertscore_precision": float(precision.mean().item() * 100),
        "bertscore_recall": float(recall.mean().item() * 100),
        "bertscore_f1": float(f1.mean().item() * 100),
        "bertscore_truncated_predictions": truncation_counts[0],
        "bertscore_truncated_references": truncation_counts[1],
    }


def compute_segmented_bertscore(predictions, references, config, device):
    from bert_score import score

    tokenizer, content_limit = load_bertscore_components(config["bertscore_model_type"])
    chunk_tokens = config["segmented_bertscore_chunk_tokens"]
    if content_limit is not None and chunk_tokens > content_limit:
        raise ValueError(f"Segment size {chunk_tokens} exceeds BERT limit {content_limit}.")

    candidates = []
    pair_references = []
    specs = []
    for prediction, reference in zip(predictions, references):
        prediction_chunks = decode_chunks(prediction, tokenizer, chunk_tokens)
        reference_chunks = decode_chunks(reference, tokenizer, chunk_tokens)
        if not prediction_chunks or not reference_chunks:
            specs.append(None)
            continue
        start = len(candidates)
        for prediction_chunk, _ in prediction_chunks:
            for reference_chunk, _ in reference_chunks:
                candidates.append(prediction_chunk)
                pair_references.append(reference_chunk)
        specs.append(
            {
                "start": start,
                "prediction_weights": [weight for _, weight in prediction_chunks],
                "reference_weights": [weight for _, weight in reference_chunks],
            }
        )

    pair_scores = []
    if candidates:
        _, _, f1 = score(
            cands=candidates,
            refs=pair_references,
            model_type=config["bertscore_model_type"],
            num_layers=config["bertscore_num_layers"],
            lang=config["bertscore_lang"],
            batch_size=config["bertscore_batch_size"],
            device=device,
            verbose=False,
        )
        pair_scores = [max(0.0, min(1.0, value)) for value in f1.tolist()]

    document_precision = []
    document_recall = []
    document_f1 = []
    for prediction, reference, spec in zip(predictions, references, specs):
        if spec is None:
            value = 1.0 if not prediction.strip() and not reference.strip() else 0.0
            document_precision.append(value)
            document_recall.append(value)
            document_f1.append(value)
            continue
        prediction_weights = spec["prediction_weights"]
        reference_weights = spec["reference_weights"]
        prediction_count = len(prediction_weights)
        reference_count = len(reference_weights)
        start = spec["start"]
        matrix = [
            pair_scores[
                start + index * reference_count:start + (index + 1) * reference_count
            ]
            for index in range(prediction_count)
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
        f1_value = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        document_precision.append(precision)
        document_recall.append(recall)
        document_f1.append(f1_value)
    denominator = max(len(predictions), 1)
    return {
        "bertscore_segmented_precision": sum(document_precision) / denominator * 100,
        "bertscore_segmented_recall": sum(document_recall) / denominator * 100,
        "bertscore_segmented_f1": sum(document_f1) / denominator * 100,
    }


def generate_for_length(model, tokenizer, data, max_new_tokens, config, device, torch):
    predictions = []
    references = []
    generated_lengths = []
    hit_count = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for index, example in enumerate(data):
        messages = build_messages(example)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config["max_seq_length"],
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": config["do_sample"],
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
        generated_tokens = output[0, inputs["input_ids"].shape[1]:]
        generated_length = int(generated_tokens.numel())
        generated_lengths.append(generated_length)
        hit_count += int(generated_length >= max_new_tokens)
        predictions.append(
            tokenizer.decode(
                generated_tokens,
                skip_special_tokens=config["skip_special_tokens"],
                clean_up_tokenization_spaces=False,
            ).strip()
        )
        references.append(str(example["output"]).strip())
        current = index + 1
        if current % config["progress_every"] == 0 or current == len(data):
            print(f"max_new_tokens={max_new_tokens}: {current}/{len(data)}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime = time.perf_counter() - start_time
    return predictions, references, generated_lengths, hit_count, runtime


def atomic_json_dump(value, path):
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def atomic_jsonl_dump(records, path):
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_path, path)


def write_summary(rows, output_dir, run_metadata):
    atomic_json_dump({"run": run_metadata, "results": rows}, os.path.join(output_dir, "summary.json"))
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with open(os.path.join(output_dir, "summary.csv"), "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = load_config()
    args = parse_args(config)
    adapter_config = validate_inputs(args)
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_eval_data(args.eval_path, args.max_samples)
    if not data:
        raise ValueError("Evaluation dataset is empty.")

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from unsloth import FastLanguageModel
    import torch

    print(f"Loading fixed adapter: {args.adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_path,
        max_seq_length=config["max_seq_length"],
        load_in_4bit=config["load_in_4bit"],
        local_files_only=True,
    )
    FastLanguageModel.for_inference(model)
    device = str(next(model.parameters()).device)
    run_metadata = {
        "adapter_path": args.adapter_path,
        "base_model_path": args.base_model_path,
        "adapter_base_model": adapter_config.get("base_model_name_or_path"),
        "eval_path": args.eval_path,
        "sample_count": len(data),
        "max_samples": args.max_samples,
        "do_sample": config["do_sample"],
        "device": device,
    }

    rows = []
    for max_new_tokens in args.lengths:
        result_path = os.path.join(args.output_dir, f"max_new_tokens_{max_new_tokens}.json")
        if os.path.exists(result_path) and not args.overwrite:
            with open(result_path, encoding="utf-8") as file:
                previous = json.load(file)
            if previous.get("run") != run_metadata:
                raise RuntimeError(
                    f"Existing result has different run metadata: {result_path}. "
                    "Use a new output directory or --overwrite."
                )
            print(f"Reuse completed result: {result_path}")
            rows.append(previous["metrics"])
            continue

        predictions, references, lengths, hit_count, runtime = generate_for_length(
            model, tokenizer, data, max_new_tokens, config, device, torch
        )
        metrics = compute_rouge(predictions, references)
        metrics["eval_bleu4"] = compute_bleu4(predictions, references)
        metrics.update(compute_bertscore(predictions, references, config, device))
        metrics.update(compute_segmented_bertscore(predictions, references, config, device))
        metrics.update(
            {
                "max_new_tokens": max_new_tokens,
                "sample_count": len(data),
                "length_limit_hit_rate": hit_count / len(data) * 100,
                "mean_generated_tokens": sum(lengths) / len(lengths),
                "generation_runtime_seconds": runtime,
            }
        )
        records = []
        for index, (prediction, reference, generated_length) in enumerate(
            zip(predictions, references, lengths)
        ):
            example = data[index]
            instruction = str(example.get("instruction", "") or "").strip()
            input_text = str(example.get("input", "") or "").strip()
            question = f"{instruction}\n{input_text}" if input_text else instruction
            records.append({
                "index": index,
                "system": str(example.get("system", "") or "").strip(),
                "instruction": instruction,
                "input": input_text,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "max_new_tokens": max_new_tokens,
                "generated_tokens": generated_length,
                "hit_length_limit": generated_length >= max_new_tokens,
            })
        atomic_json_dump(
            {"run": run_metadata, "metrics": metrics, "predictions": records},
            result_path,
        )
        atomic_jsonl_dump(
            records,
            os.path.join(
                args.output_dir,
                f"predictions_max_new_tokens_{max_new_tokens}.jsonl",
            ),
        )
        rows.append(metrics)
        write_summary(rows, args.output_dir, run_metadata)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    rows.sort(key=lambda row: row["max_new_tokens"])
    write_summary(rows, args.output_dir, run_metadata)
    print(f"Finished. Summary: {os.path.join(args.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
