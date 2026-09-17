"""Evaluate a Bailian OpenAI-compatible model with the local baseline protocol."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import bailian_config as config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bailian inference aligned with constrast_infer local baselines."
    )
    parser.add_argument("--model", default=config.BAILIAN_MODEL)
    parser.add_argument("--base-url", default=config.BAILIAN_BASE_URL)
    parser.add_argument("--eval-path", default=config.EVAL_DATASET_PATH)
    parser.add_argument("--output-path", default=config.OUTPUT_PATH)
    parser.add_argument("--max-samples", type=int, default=config.MAX_SAMPLES)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=config.TOP_P)
    parser.add_argument("--progress-every", type=int, default=config.PROGRESS_EVERY)
    parser.add_argument("--timeout", type=float, default=config.REQUEST_TIMEOUT)
    parser.add_argument("--retries", type=int, default=config.MAX_RETRIES)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=3.0,
        help="Seconds to wait between samples. Use a positive value for remote APIs.",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Stop early after this many failed samples to preserve a resumable run.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=config.RESUME,
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=config.ENABLE_THINKING,
    )
    return parser.parse_args()


def safe_text(value):
    return "" if value is None else str(value)


def build_messages(example):
    instruction = safe_text(example.get(config.INSTRUCTION_COLUMN)).strip()
    input_text = safe_text(example.get(config.INPUT_COLUMN)).strip()
    reference = safe_text(example.get(config.OUTPUT_COLUMN)).strip()
    system_text = safe_text(example.get(config.SYSTEM_COLUMN)).strip()

    user_query = instruction
    if input_text:
        user_query = f"{instruction}\n{input_text}" if instruction else input_text

    messages = []
    if config.INCLUDE_SYSTEM_PROMPT and system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_query})
    return messages, reference


def read_json_records(path):
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", "records", "train"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError(f"Unsupported JSON dataset structure: {path}")


def partial_path_for(output_path):
    path = Path(output_path)
    suffix = path.suffix or ".json"
    return path.with_suffix(f"{suffix}.partial.jsonl")


def load_completed(partial_path):
    completed = {}
    if not partial_path.exists():
        return completed
    with partial_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("status") == "success":
                    completed[int(record["index"])] = record
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                print(f"Ignore invalid partial line {line_number}: {error}")
    return completed


def append_partial(partial_path, record):
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def request_one(api_key, args, messages):
    request = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "stream": False,
        "extra_body": {"enable_thinking": args.enable_thinking},
    }
    if args.top_p is not None:
        request["top_p"] = args.top_p

    last_error = None
    for attempt in range(args.retries + 1):
        client = None
        try:
            # Create a fresh client for each attempt. Some OpenAI-compatible
            # providers close an idle/reused HTTP connection during long runs.
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=args.base_url,
                timeout=args.timeout,
                max_retries=0,
            )
            response = client.chat.completions.create(**request)
            content = response.choices[0].message.content
            if isinstance(content, list):
                content = "".join(
                    safe_text(item.get("text") if isinstance(item, dict) else item)
                    for item in content
                )
            content = safe_text(content).strip()
            if not content:
                raise ValueError("Bailian returned an empty answer.")
            usage = response.usage.model_dump() if response.usage is not None else None
            return content, usage, getattr(response, "id", None)
        except Exception as error:
            last_error = error
            if attempt >= args.retries:
                break
            wait_seconds = config.RETRY_BACKOFF_SECONDS * (2 ** attempt)
            print(
                f"Request failed ({attempt + 1}/{args.retries + 1}, "
                f"{type(error).__name__}): {error}; "
                f"retry in {wait_seconds:.1f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
        finally:
            if client is not None:
                client.close()
    raise RuntimeError(str(last_error)) from last_error


def calculate_metrics(predictions, references):
    # Import after CUDA_VISIBLE_DEVICES is set. Reusing these functions guarantees
    # exactly the same ROUGE/BLEU/BERTScore implementation as local baselines.
    os.environ["INFER_METRICS_ONLY"] = "1"
    import infer_common as shared

    shared.cfg = config
    rouge = shared.compute_average_rouge(predictions, references)
    metrics = {
        "eval_rouge1": rouge["rouge1"],
        "eval_rouge2": rouge["rouge2"],
        "eval_rougeL": rouge["rougeL"],
        "eval_bleu4": shared.compute_corpus_bleu(
            predictions,
            references,
            max_order=config.BLEU_MAX_ORDER,
            smooth_value=config.BLEU_SMOOTH_VALUE,
        ),
    }

    device = "cuda:0" if shared.torch.cuda.is_available() else "cpu"
    print(f"Start aligned BERTScore on {len(predictions)} samples; device={device}")
    bertscore = shared.compute_average_bertscore(
        predictions,
        references,
        model_type=config.BERTSCORE_MODEL_TYPE,
        num_layers=config.BERTSCORE_NUM_LAYERS,
        lang=config.BERTSCORE_LANG,
        batch_size=config.BERTSCORE_BATCH_SIZE,
        device=device,
    )
    metrics.update({f"eval_{key}": value for key, value in bertscore.items()})
    segmented = shared.compute_segmented_bertscore(
        predictions,
        references,
        model_type=config.BERTSCORE_MODEL_TYPE,
        num_layers=config.BERTSCORE_NUM_LAYERS,
        lang=config.BERTSCORE_LANG,
        batch_size=config.BERTSCORE_BATCH_SIZE,
        chunk_tokens=config.SEGMENTED_BERTSCORE_CHUNK_TOKENS,
        device=device,
    )
    metrics.update({f"eval_{key}": value for key, value in segmented.items()})
    return metrics


def main():
    args = parse_args()
    api_key = os.getenv(config.API_KEY_ENV, "").strip()
    if not api_key:
        raise ValueError(
            f"Missing API key. Export {config.API_KEY_ENV} before running."
        )
    if "bailian.console.aliyun.com" in args.base_url:
        raise ValueError("base_url is a console webpage, not a model API endpoint.")

    dataset = read_json_records(args.eval_path)
    total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    if total <= 0:
        raise ValueError("No evaluation samples selected.")

    output_path = Path(args.output_path).resolve()
    partial_path = partial_path_for(output_path)
    if not args.resume and partial_path.exists():
        raise FileExistsError(
            f"Partial file exists: {partial_path}; use --resume or choose a new output path."
        )
    completed = load_completed(partial_path) if args.resume else {}
    completed = {index: record for index, record in completed.items() if index < total}
    compatible_completed = {}
    for index, record in completed.items():
        expected_messages, _ = build_messages(dataset[index])
        if record.get("prompt_messages") == expected_messages:
            compatible_completed[index] = record
    ignored_partial_records = len(completed) - len(compatible_completed)
    completed = compatible_completed

    print(f"model = {args.model}")
    print(f"base_url = {args.base_url}")
    print(f"eval_path = {args.eval_path}")
    print(f"output_path = {output_path}")
    print(f"partial_path = {partial_path}")
    print(
        f"samples = {total}; resumed = {len(completed)}; "
        f"ignored_incompatible_partial = {ignored_partial_records}"
    )
    print(
        "prompt_protocol = "
        + ("system+user" if config.INCLUDE_SYSTEM_PROMPT else "user-only")
    )
    print(
        f"generation = max_new_tokens={args.max_new_tokens}, "
        f"temperature={args.temperature}, thinking={args.enable_thinking}"
    )

    failures = []
    consecutive_failures = 0
    for index in range(total):
        if index in completed:
            continue
        messages, reference = build_messages(dataset[index])
        try:
            prediction, usage, response_id = request_one(api_key, args, messages)
            record = {
                "index": index,
                "status": "success",
                "prompt_messages": messages,
                "prompt": json.dumps(messages, ensure_ascii=False),
                "reference": reference,
                "prediction": prediction,
                "response_id": response_id,
                "usage": usage,
            }
            append_partial(partial_path, record)
            completed[index] = record
            consecutive_failures = 0
        except Exception as error:
            consecutive_failures += 1
            failures.append({"index": index, "error": str(error)})
            print(
                f"Sample {index} failed ({type(error).__name__}): {error}; "
                f"consecutive_failures={consecutive_failures}",
                flush=True,
            )
            if consecutive_failures >= args.max_consecutive_failures:
                print(
                    "Stopping after consecutive failures; successful records are "
                    "preserved. Retry later with --resume.",
                    flush=True,
                )
                break

        processed = index + 1
        if processed % args.progress_every == 0 or processed == total:
            print(
                f"API generation progress: {processed}/{total}; "
                f"success={len(completed)}, failed={len(failures)}",
                flush=True,
            )
        if args.request_interval > 0 and index + 1 < total:
            time.sleep(args.request_interval)

    ordered_records = [completed[index] for index in range(total) if index in completed]
    if len(ordered_records) != total:
        raise RuntimeError(
            f"Only {len(ordered_records)}/{total} samples succeeded. "
            f"Run the same command with --resume to retry missing samples."
        )

    predictions = [record["prediction"] for record in ordered_records]
    references = [record["reference"] for record in ordered_records]
    metrics = calculate_metrics(predictions, references)

    result = {
        "provider": "Alibaba Cloud Model Studio (Bailian)",
        "model_path": args.model,
        "adapter_path": None,
        "base_url": args.base_url,
        "eval_path": args.eval_path,
        "num_samples": len(ordered_records),
        "max_new_tokens": args.max_new_tokens,
        "decode_strategy": "temperature_0",
        "enable_thinking": args.enable_thinking,
        "prompt_protocol": (
            "system+user" if config.INCLUDE_SYSTEM_PROMPT else "user-only"
        ),
        "metric_protocol": "constrast_infer_local_baseline_aligned_v1",
        "metrics": metrics,
        "records": ordered_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            ensure_ascii=config.output_ensure_ascii,
            indent=config.output_indent,
        )
    print(f"Saved aligned result: {output_path}")
    for key, value in metrics.items():
        print(f"{key} = {value:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES",
        os.getenv("INFER_CUDA_VISIBLE_DEVICES", config.DEFAULT_CUDA_VISIBLE_DEVICES),
    )
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed API responses remain in the partial JSONL.")
        sys.exit(130)
