"""Batch-evaluate a Dify RAG application on the railway QA validation set."""

import argparse
import csv
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from functools import lru_cache


DEFAULT_DATA_DIR = "/data16T/wjy/Learn_llm/铁路大模型/dataset_splits"
DEFAULT_OUTPUT_DIR = "/data16T/wjy/Learn_llm/铁路大模型/new_7b/rag_eval/outputs"
DEFAULT_BASE_URL = "http://127.0.0.1:18081/v1"
DEFAULT_BERTSCORE_MODEL = "/data16T/wjy/models/Bert"
METRIC_PROTOCOL = "new_7b_unified_generation_metrics_v1_20260823"


def load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def tokenize_for_rouge(text):
    """Match new_7b: character-level tokens, preserving punctuation and case."""
    return [char for char in str(text).strip() if not char.isspace()]


def rouge_n_f1(prediction_tokens, reference_tokens, n):
    if len(prediction_tokens) < n or len(reference_tokens) < n:
        return 0.0
    prediction_ngrams = Counter(
        tuple(prediction_tokens[index:index + n])
        for index in range(len(prediction_tokens) - n + 1)
    )
    reference_ngrams = Counter(
        tuple(reference_tokens[index:index + n])
        for index in range(len(reference_tokens) - n + 1)
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


def compute_average_rouge(predictions, references):
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for prediction, reference in zip(predictions, references):
        prediction_tokens = tokenize_for_rouge(prediction)
        reference_tokens = tokenize_for_rouge(reference)
        scores["rouge1"].append(rouge_n_f1(prediction_tokens, reference_tokens, 1))
        scores["rouge2"].append(rouge_n_f1(prediction_tokens, reference_tokens, 2))
        scores["rougeL"].append(rouge_l_f1(prediction_tokens, reference_tokens))
    denominator = max(len(predictions), 1)
    return {key: sum(values) / denominator * 100 for key, values in scores.items()}


def compute_corpus_bleu4(predictions, references):
    """Match new_7b: single-reference corpus BLEU-4 without smoothing."""
    if not predictions or not references:
        return 0.0
    max_order = 4
    matches = [0] * max_order
    possible = [0] * max_order
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
            matches[order - 1] += sum((prediction_ngrams & reference_ngrams).values())
            possible[order - 1] += sum(prediction_ngrams.values())
    if prediction_length == 0:
        return 0.0
    precisions = []
    for matched, candidate_count in zip(matches, possible):
        if candidate_count == 0:
            continue
        precision = matched / candidate_count
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


def build_query(item, include_system=False):
    parts = []
    if include_system and item.get("system"):
        parts.append(str(item["system"]).strip())
    if item.get("instruction"):
        parts.append(str(item["instruction"]).strip())
    if item.get("input"):
        parts.append(str(item["input"]).strip())
    return "\n\n".join(part for part in parts if part)


def call_dify(base_url, api_key, query, user, timeout, retries, sleep_seconds):
    url = base_url.rstrip("/") + "/chat-messages"
    payload = {
        "inputs": {},
        "query": query,
        "response_mode": "blocking",
        "conversation_id": "",
        "user": user,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    total_attempts = retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            return data, None
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            last_error = f"HTTP {e.code}: {detail}"
        except Exception as e:
            last_error = str(e)

        if attempt < total_attempts:
            time.sleep(sleep_seconds)

    return None, last_error


def extract_answer(dify_response):
    if not isinstance(dify_response, dict):
        return ""
    for key in ["answer", "text", "result", "output"]:
        value = dify_response.get(key)
        if isinstance(value, str):
            return value
    return json.dumps(dify_response, ensure_ascii=False)


def evaluate_rouge(reference, prediction):
    """Per-sample values are retained for error analysis; summary uses macro means."""
    prediction_tokens = tokenize_for_rouge(prediction)
    reference_tokens = tokenize_for_rouge(reference)
    return {
        "rouge1_f1": round(rouge_n_f1(prediction_tokens, reference_tokens, 1) * 100, 6),
        "rouge2_f1": round(rouge_n_f1(prediction_tokens, reference_tokens, 2) * 100, 6),
        "rouge_l_f1": round(rouge_l_f1(prediction_tokens, reference_tokens) * 100, 6),
    }


@lru_cache(maxsize=4)
def load_bertscore_tokenizer_and_limit(model_type):
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_type, local_files_only=True, use_fast=True)
    config = AutoConfig.from_pretrained(model_type, local_files_only=True)
    max_positions = getattr(config, "max_position_embeddings", None)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if tokenizer_limit is not None and tokenizer_limit > 100000:
        tokenizer_limit = None
    raw_limit = max_positions if max_positions is not None else tokenizer_limit
    if raw_limit is None:
        return tokenizer, None
    content_limit = max(int(raw_limit) - tokenizer.num_special_tokens_to_add(pair=False), 1)
    return tokenizer, content_limit


def truncate_texts_for_bertscore(texts, tokenizer, max_content_tokens):
    if max_content_tokens is None:
        return texts, 0, 0
    output = []
    truncated_count = 0
    longest = 0
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        longest = max(longest, len(token_ids))
        if len(token_ids) > max_content_tokens:
            token_ids = token_ids[:max_content_tokens]
            truncated_count += 1
        output.append(tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return output, truncated_count, longest


def split_text_for_segmented_bertscore(text, tokenizer, chunk_tokens):
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


def compute_segmented_bertscore(
    predictions, references, model_type, num_layers, batch_size, chunk_tokens
):
    from bert_score import score

    tokenizer, max_content_tokens = load_bertscore_tokenizer_and_limit(model_type)
    if max_content_tokens is not None and chunk_tokens > max_content_tokens:
        raise ValueError(f"segmented chunk size {chunk_tokens} exceeds {max_content_tokens}")
    pair_candidates = []
    pair_references = []
    specs = []
    for prediction, reference in zip(predictions, references):
        prediction_chunks = split_text_for_segmented_bertscore(prediction, tokenizer, chunk_tokens)
        reference_chunks = split_text_for_segmented_bertscore(reference, tokenizer, chunk_tokens)
        if not prediction_chunks or not reference_chunks:
            specs.append(None)
            continue
        start = len(pair_candidates)
        for prediction_chunk, _ in prediction_chunks:
            for reference_chunk, _ in reference_chunks:
                pair_candidates.append(prediction_chunk)
                pair_references.append(reference_chunk)
        specs.append({
            "start": start,
            "prediction_weights": [weight for _, weight in prediction_chunks],
            "reference_weights": [weight for _, weight in reference_chunks],
        })
    pair_scores = []
    if pair_candidates:
        _, _, f1 = score(
            pair_candidates,
            pair_references,
            lang="zh",
            model_type=model_type,
            num_layers=num_layers,
            batch_size=batch_size,
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
            pair_scores[start + index * reference_count:start + (index + 1) * reference_count]
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
        "eval_bertscore_segmented_precision": sum(document_precision) / denominator * 100,
        "eval_bertscore_segmented_recall": sum(document_recall) / denominator * 100,
        "eval_bertscore_segmented_f1": sum(document_f1) / denominator * 100,
    }


def is_success_row(row):
    """Only a non-error row with a non-empty prediction is a successful result."""
    return not row.get("error") and bool(str(row.get("prediction", "")).strip())


def add_unified_bertscore(rows, model_type, num_layers, batch_size, chunk_tokens):
    success_rows = [row for row in rows if is_success_row(row)]
    if not success_rows:
        return {}

    try:
        from bert_score import score
    except ImportError as e:
        raise ImportError(
            "BERTScore运行依赖加载失败。请使用 rag_eval/run_batch_dify_rag_eval_unified.sh "
            f"启动；原始错误：{e}"
        ) from e

    original_predictions = [row.get("prediction", "") for row in success_rows]
    original_references = [row.get("reference", "") for row in success_rows]
    tokenizer, content_limit = load_bertscore_tokenizer_and_limit(model_type)
    predictions, truncated_predictions, max_prediction_length = truncate_texts_for_bertscore(
        original_predictions, tokenizer, content_limit
    )
    references, truncated_references, max_reference_length = truncate_texts_for_bertscore(
        original_references, tokenizer, content_limit
    )

    precision, recall, f1 = score(
        predictions,
        references,
        lang="zh",
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        verbose=False,
    )

    for row, p, r, f in zip(success_rows, precision, recall, f1):
        row["bertscore_precision"] = round(float(p) * 100, 6)
        row["bertscore_recall"] = round(float(r) * 100, 6)
        row["bertscore_f1"] = round(float(f) * 100, 6)

    metrics = {
        "eval_bertscore_precision": float(precision.mean().item() * 100),
        "eval_bertscore_recall": float(recall.mean().item() * 100),
        "eval_bertscore_f1": float(f1.mean().item() * 100),
        "bertscore_content_token_limit": content_limit,
        "bertscore_truncated_predictions": truncated_predictions,
        "bertscore_truncated_references": truncated_references,
        "bertscore_max_prediction_tokens": max_prediction_length,
        "bertscore_max_reference_tokens": max_reference_length,
    }
    metrics.update(compute_segmented_bertscore(
        original_predictions,
        original_references,
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        chunk_tokens=chunk_tokens,
    ))
    return metrics


def load_latest_rows(jsonl_path):
    """Load the latest durable result for each sample from an append-only JSONL."""
    latest_rows = {}
    if not os.path.exists(jsonl_path):
        return latest_rows

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = item.get("id")
            if sample_id is not None:
                latest_rows[sample_id] = item
    return latest_rows


def append_jsonl(path, item):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    fieldnames = [
        "id",
        "question",
        "reference",
        "prediction",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "rouge1_f1",
        "rouge2_f1",
        "rouge_l_f1",
        "error",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def summarize(rows, bertscore_metrics):
    success_rows = [row for row in rows if is_success_row(row)]
    total = len(rows)
    success = len(success_rows)
    failed = total - success
    predictions = [row.get("prediction", "") for row in success_rows]
    references = [row.get("reference", "") for row in success_rows]
    rouge = compute_average_rouge(predictions, references)
    summary = {
        "total": total,
        "success": success,
        "failed": failed,
        "eval_rouge1": rouge["rouge1"],
        "eval_rouge2": rouge["rouge2"],
        "eval_rougeL": rouge["rougeL"],
        "eval_bleu4": compute_corpus_bleu4(predictions, references),
        "metric_scale": "0-100",
        "tokenization": "character-level, ignore whitespace, preserve punctuation and case",
        "bleu_smoothing": 0.0,
    }
    summary.update(bertscore_metrics)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="批量调用 Dify RAG 平台，并根据 testdata 标准答案评估结果。")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="数据文件所在目录")
    parser.add_argument("--test-file", default="", help="测试集文件名；不填时优先使用 testdata799.json")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Dify API 基础地址，例如 http://127.0.0.1:18081/v1")
    parser.add_argument("--api-key", default=os.getenv("DIFY_API_KEY", ""), help="Dify API Key；也可用环境变量 DIFY_API_KEY")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="结果输出目录")
    parser.add_argument("--run-name", default="", help="固定运行名称；使用 --resume 时必须传入同一个名称")
    parser.add_argument("--user", default="batch-eval-user", help="Dify 请求中的 user 标识")
    parser.add_argument("--limit", type=int, default=0, help="只测试前 N 条；0 表示全量")
    parser.add_argument("--start", type=int, default=0, help="从第几条开始，0 表示从头开始")
    parser.add_argument("--timeout", type=int, default=120, help="单条请求超时时间，单位秒")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="首次请求失败后的重试次数；0 表示请求一次但不重试",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="每次请求后的等待秒数")
    parser.add_argument("--include-system", action="store_true", help="把 system 字段一起拼到问题前面")
    parser.add_argument("--resume", action="store_true", help="断点续跑：跳过已经成功写入 JSONL 的样本")
    parser.add_argument("--bertscore-model", default=DEFAULT_BERTSCORE_MODEL, help="BERTScore 使用的本地模型")
    parser.add_argument("--bertscore-num-layers", type=int, default=12, help="BERTScore 模型层数")
    parser.add_argument("--bertscore-batch-size", type=int, default=8, help="BERTScore 批量计算大小")
    parser.add_argument("--segmented-chunk-tokens", type=int, default=450, help="分段 BERTScore 每段 token 数")
    return parser.parse_args()


def validate_metric_runtime():
    """Fail before API calls when PyTorch native CUDA libraries are unavailable."""
    try:
        import torch
        import bert_score  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "评测依赖加载失败，尚未调用Dify。请改用 "
            "rag_eval/run_batch_dify_rag_eval_unified.sh 启动。"
            f" 原始错误：{error}"
        ) from error
    print(f"评测运行环境检查通过: torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")


def main():
    args = parse_args()
    if not args.api_key:
        raise ValueError("缺少 API Key。请使用 --api-key，或先设置环境变量 DIFY_API_KEY。")
    if args.resume and not args.run_name:
        raise ValueError("使用 --resume 时必须通过 --run-name 指定要继续的运行。")
    if args.retries < 0:
        raise ValueError("--retries 不能小于 0。")
    validate_metric_runtime()

    test_file = args.test_file
    if not test_file:
        test_file = "testdata799.json"


    test_path = os.path.join(args.data_dir, test_file)
    test_data = load_json_list(test_path)

    selected_data = test_data[args.start:]
    if args.limit > 0:
        selected_data = selected_data[:args.limit]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(output_dir, f"dify_unified_eval_details_{run_name}.jsonl")
    raw_jsonl_path = os.path.join(output_dir, f"dify_unified_eval_raw_{run_name}.jsonl")
    csv_path = os.path.join(output_dir, f"dify_unified_eval_details_{run_name}.csv")
    summary_path = os.path.join(output_dir, f"dify_unified_eval_summary_{run_name}.json")

    latest_rows = load_latest_rows(raw_jsonl_path) if args.resume else {}
    finished_ids = {
        sample_id
        for sample_id, row in latest_rows.items()
        if is_success_row(row)
    }
    rows_by_id = dict(latest_rows)
    processed_count = 0

    print(f"测试文件: {test_path}")
    print(f"计划评估: {len(selected_data)} 条")
    print(f"Dify 地址: {args.base_url.rstrip('/') + '/chat-messages'}")

    for offset, item in enumerate(selected_data, start=args.start):
        sample_id = offset
        if args.resume and sample_id in finished_ids:
            continue

        question = build_query(item, include_system=args.include_system)
        reference = item.get("output", "")
        dify_response, error = call_dify(
            base_url=args.base_url,
            api_key=args.api_key,
            query=question,
            user=args.user,
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep,
        )

        if error:
            row = {
                "id": sample_id,
                "question": question,
                "reference": reference,
                "prediction": "",
                "error": error,
            }
        else:
            prediction = extract_answer(dify_response).strip()
            if not prediction:
                row = {
                    "id": sample_id,
                    "question": question,
                    "reference": reference,
                    "prediction": "",
                    "dify_response": dify_response,
                    "error": "Dify returned an empty answer",
                }
            else:
                scores = evaluate_rouge(reference, prediction)
                row = {
                    "id": sample_id,
                    "question": question,
                    "reference": reference,
                    "prediction": prediction,
                    "dify_response": dify_response,
                    "error": "",
                    **scores,
                }

        rows_by_id[sample_id] = row
        append_jsonl(raw_jsonl_path, row)

        processed_count += 1
        if processed_count == 1 or processed_count % 10 == 0:
            print(f"本次已处理 {processed_count} 条")

        time.sleep(args.sleep)

    selected_ids = set(range(args.start, args.start + len(selected_data)))
    rows = [
        rows_by_id[sample_id]
        for sample_id in sorted(selected_ids)
        if sample_id in rows_by_id
    ]

    print(f"开始计算统一口径 BERTScore，共汇总 {len(rows)} 条结果")
    bertscore_metrics = add_unified_bertscore(
        rows,
        model_type=args.bertscore_model,
        num_layers=args.bertscore_num_layers,
        batch_size=args.bertscore_batch_size,
        chunk_tokens=args.segmented_chunk_tokens,
    )

    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    summary = summarize(rows, bertscore_metrics)
    summary["test_file"] = test_path
    summary["metric_protocol"] = METRIC_PROTOCOL
    summary["bertscore_model"] = args.bertscore_model
    summary["bertscore_num_layers"] = args.bertscore_num_layers
    summary["segmented_bertscore_chunk_tokens"] = args.segmented_chunk_tokens
    summary["raw_jsonl"] = raw_jsonl_path
    summary["details_jsonl"] = jsonl_path
    summary["details_csv"] = csv_path

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("评估完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
