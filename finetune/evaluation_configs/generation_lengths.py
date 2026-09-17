"""Inference-only generation-length ablation for one fixed trained adapter."""

CONFIG = {
    "base_model_path": "/data16T/wjy/models/Qwen2.5-7B-Instruct",
    # Keep this as None in source control and pass --adapter-path explicitly.
    # This prevents accidentally evaluating a stale or base-only model.
    "adapter_path": None,
    "eval_dataset_path": "/data16T/wjy/Learn_llm/铁路大模型/dataset_splits/testdata799.json",
    "output_dir": "/data16T/wjy/Learn_llm/铁路大模型/new_7b/evaluation_outputs/generation_lengths",
    "max_new_tokens_values": [128, 256, 512, 1024, 1536],
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "do_sample": False,
    "skip_special_tokens": True,
    "progress_every": 10,
    "bertscore_model_type": "/data16T/wjy/models/Bert",
    "bertscore_num_layers": 12,
    "bertscore_lang": "zh",
    "bertscore_batch_size": 8,
    "segmented_bertscore_chunk_tokens": 450,
    "metrics": [
        "rouge1",
        "rouge2",
        "rougeL",
        "eval_bleu4",
        "bertscore_f1",
        "bertscore_segmented_f1",
        "length_limit_hit_rate",
        "mean_generated_tokens",
        "generation_runtime_seconds",
    ],
}
