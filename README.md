# RailOpsLLM

## A Domain-Adaptive Large Language Model for Railway Information System Operation and Maintenance via Fine-Tuning and Retrieval-Augmented Generation

<p align="center">
  <img src="assets/railopsllm-overview.png" alt="RailOpsLLM system overview" width="920">
</p>

<p align="center"><em>Figure 1. Overview of the RailOpsLLM framework.</em></p>

Reproducibility code and metric-only artifacts for railway-domain question answering.

## Scope and data policy

This repository contains fine-tuning, Dify RAG evaluation and direct-inference code. It deliberately does **not** distribute training data, test data, railway knowledge-base text, API keys, raw prompts, predictions, traces, runtime logs or base-model weights.

The final LoRA adapter is available from the GitHub Release. It must be loaded on `Qwen/Qwen2.5-7B-Instruct`.

## Layout

- `finetune/`: final LoRA training configuration and launcher.
- `rag/`: batch Dify RAG evaluation client.
- `base_inference/`: local and Bailian-compatible direct inference.
- `results/`: metric-only summaries and selected checkpoint metadata.
- `model/`: adapter metadata and release manifest.

## Environment

Create an environment from `environment.yml` or install `requirements.txt`. Activate an environment containing PyTorch, Unsloth, PEFT, Transformers and BERTScore before running scripts.

Set paths for private assets; none are included in this repository:

```bash
export MODEL_ROOT=/path/to/models
export TRAIN_DATASET_PATH=/path/to/traindata7236.json
export EVAL_DATASET_PATH=/path/to/testdata799.json
export BERTSCORE_MODEL=/path/to/local/bert-model
```

## Fine-tuning

```bash
bash finetune/run_train.sh finetune/train_config_final.py
```

The final configuration is LoRA r=64, alpha=64, dropout=0, learning rate 3e-4, warmup=50, epoch=2, effective batch size=2, sequence length=2048, linear scheduler and seed=42.

## Dify RAG evaluation

```bash
export DIFY_API_KEY='your-key'
export DIFY_BASE_URL='http://your-dify-host:port/v1'
export RAG_DATA_DIR=/path/to/private-data
export RAG_OUTPUT_DIR=outputs/rag
bash rag/run_batch_dify_rag_eval_unified.sh --test-file "$EVAL_DATASET_PATH" --run-name example
```

## Direct inference

For Bailian-compatible APIs:

```bash
export DASHSCOPE_API_KEY='your-key'
export EVAL_DATASET_PATH=/path/to/testdata799.json
python3 base_inference/run_bailian_infer.py --help
```

For local Qwen2.5 inference:

```bash
export MODEL_ROOT=/path/to/models
export EVAL_DATASET_PATH=/path/to/testdata799.json
bash base_inference/run_infer_qwen25_7b_eval.sh --fg
```

## Release integrity

Verify the release archive with its accompanying `.sha256` file. The final adapter was selected at checkpoint-7200 with segmented BERTScore F1 79.76306056835053.

## Licence

No licence has yet been selected. Contact the repository owner before reusing this code or any derivative artifact.
