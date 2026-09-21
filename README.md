# Rail|OpsLLM

## A Domain-Adaptive Large Language Model for Railway Information System Operation and Maintenance via Fine-Tuning and Retrieval-Augmented Generation

<p align="center">
  <a href="https://github.com/1wangjy/railway-llm-rag">Code</a> |
  <a href="results/">Results</a>
</p>

## Abstract

Rail|OpsLLM is a domain-adaptive large language model project for question answering in railway information-system operation and maintenance (O&M). It combines parameter-efficient LoRA fine-tuning of `Qwen/Qwen2.5-7B-Instruct` with retrieval-augmented generation (RAG), and provides reproducible code for fine-tuning, Dify-based RAG evaluation, direct base-model inference, and unified metric reporting.

This public repository contains desensitized code, model metadata, aggregate dataset statistics, selected evaluation summaries, and the final LoRA adapter release. It does not distribute private railway documents, raw instruction data, knowledge-base content, credentials, prompts, predictions, traces, runtime logs, or base-model weights.

## Highlights

- Domain adaptation for railway information-system O&M using PEFT LoRA.
- Dify-based RAG evaluation over a private railway knowledge base.
- Direct-inference comparison across ten local and API-accessed language models.
- Unified evaluation on a held-out set of 799 samples using ROUGE, BLEU-4, and BERTScore variants.
- Final adapter selected at checkpoint 7200 with recorded segmented BERTScore F1 of `79.7631`.

<!--
## Latest release

- **Railway Qwen2.5-7B LoRA v1.0.1**: [download the adapter and checksums](https://github.com/1wangjy/railway-llm-rag/releases/tag/v1.0.1).
- Base model: [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
- The release contains the LoRA adapter and tokenizer assets, not the base model or private datasets.
-->

## Contents

- [Framework](#framework)
- [Dataset composition](#dataset-composition)
- [Experimental results](#experimental-results)
- [Fine-tuning results](#fine-tuning-results)
- [RAG results](#rag-results)
- [Base-model direct-inference results](#base-model-direct-inference-results)
- [Repository layout](#repository-layout)
- [Environment](#environment)
- [Fine-tuning](#fine-tuning)
- [Dify RAG evaluation](#dify-rag-evaluation)
- [Direct inference](#direct-inference)
- [Data privacy and safety statement](#data-privacy-and-safety-statement)
- [Licence](#licence)

## Framework

<p align="center">
  <img src="assets/railopsllm-overview.png" alt="Rail|OpsLLM system overview" width="920">
</p>

<p align="center"><em>Figure 1. Overview of the Rail|OpsLLM framework.</em></p>

## Dataset composition

The railway O&M instruction dataset contains 27,326 instruction-response pairs across five knowledge categories. Raw documents and instruction data are not distributed in this repository.

<p align="center">
  <img src="assets/Table%201-Data%20category.png"
       alt="Composition of the railway O&M instruction dataset"
       width="920">
</p>

<p align="center">
  <em>Table 1. Source composition of the railway O&M instruction dataset.</em>
</p>

### Chinese version

<p align="center">
  <img src="assets/table-6-1-dataset-composition-cn.png"
       alt="表 6.1 各类型指令微调数据集统计表"
       width="920">
</p>

<p align="center">
  <em>表 6.1 各类型指令微调数据集统计表。</em>
</p>

Only aggregate dataset statistics are reported for reproducibility and privacy protection. See [Data privacy and safety statement](#data-privacy-and-safety-statement) for the release policy and usage restrictions.

## Experimental results

All reported scores use a `0-100` scale. The private evaluation set contains 799 samples. Predictions and references are not distributed.

### Fine-tuning results

| Model | Adapter | Max new tokens | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | BERTScore F1 | Segmented BERTScore F1 |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | No | 1024 | 35.6663 | 17.4696 | 18.6856 | 11.4633 | 73.5215 | 72.8931 |
| Rail|OpsLLM LoRA (`checkpoint-7200`) | Yes | 256 | 52.8702 | 35.0510 | 36.7110 | 13.8758 | 80.2740 | **79.8591** |
| Rail|OpsLLM LoRA (`checkpoint-7200`) | Yes | 512 | 55.6769 | 36.1768 | 37.2475 | 20.2500 | 80.3624 | 79.7227 |
| Rail|OpsLLM LoRA (`checkpoint-7200`) | Yes | 1024 | **56.1948** | **36.4229** | **37.3637** | **22.4277** | **80.3624** | 79.7132 |

Complete generation-length results are available in [`results/finetune/generation_metrics.json`](results/finetune/generation_metrics.json).

### RAG results

| Model / workflow | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | BERTScore P | BERTScore R | BERTScore F1 | Seg. P | Seg. R | Seg. F1 | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-32B | 42.3890 | 24.2712 | 27.4604 | 10.8935 | 73.9407 | 74.8593 | 74.1369 | 74.0429 | 73.6858 | 73.8603 | Complete |
| Qwen3.7-Plus | 33.2360 | 21.4453 | 24.2434 | 8.6570 | 76.0328 | 68.5623 | 71.4814 | 71.2679 | 71.0126 | 71.1331 | Complete |
| ChatGLM3-6B | 34.1442 | 19.8104 | 22.9115 | 8.9152 | 72.0726 | 71.1778 | 71.2978 | 71.2829 | 70.9047 | 71.0888 | Complete; resumed 83 samples |
| Llama-3.1-8B-Instruct | 33.6550 | 16.1281 | 19.6592 | 9.3309 | 69.4651 | 70.6718 | 69.8523 | 69.8042 | 69.4261 | 69.6122 | Complete |
| Qwen3-8B | 30.4191 | 15.6058 | 16.5670 | 8.0385 | 70.3057 | 77.5017 | 73.4518 | 71.9579 | 73.2378 | 72.5823 | Complete |
| DeepSeek-R1-Distill-Qwen-7B | 1.2155 | 0.3360 | 0.9974 | 0.1240 | 46.9599 | 46.9337 | 46.9223 | 47.0011 | 47.1079 | 47.0526 | Complete |
| DeepSeek-V4-Pro-0813 | 45.7284 | 34.7586 | 38.0347 | 4.5094 | 83.4941 | 75.2203 | 78.8318 | 78.9359 | 78.3018 | 78.6120 | Complete; no prediction truncation |
| GLM-5 | 39.6198 | 28.7304 | 31.5458 | 4.9146 | 81.3714 | 72.4507 | 76.1688 | 76.2903 | 75.6315 | 75.9529 | Complete |
| Qwen2.5-7B-Instruct | 44.2146 | 31.1207 | 33.9795 | 7.5883 | 80.6745 | 75.7561 | 77.7918 | 77.8705 | 77.3124 | 77.5857 | Complete |
| Qwen2.5-7B-Instruct-SFT | **54.1948** | **39.4338** | **41.4979** | **12.5453** | **83.2301** | **80.6536** | **81.6662** | **81.6899** | **80.9872** | **81.3308** | Complete |

The full ten-model RAG comparison is documented in [`results/rag/README.md`](results/rag/README.md), with machine-readable metrics in [`results/rag/model_comparison_full799.json`](results/rag/model_comparison_full799.json).

### Base-model direct-inference results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | BERTScore F1 | Segmented BERTScore F1 | Metric protocol |
|---|---:|---:|---:|---:|---:|---:|---|
| ChatGLM3-6B | **35.8651** | **18.4609** | 21.8958 | **12.9917** | 73.2305 | 72.8975 | `constrast_infer_char_level_v2_20260825` |
| Qwen2.5-7B-Instruct | 35.6663 | 17.4696 | 18.6856 | 11.4633 | 73.5215 | 72.8931 | `constrast_infer_char_level_v2_20260825` |
| DeepSeek-LLM-7B-Chat | 34.5345 | 17.6410 | 20.9668 | 12.1948 | **73.5426** | **73.0899** | `constrast_infer_char_level_v2_20260825` |
| Llama-3.1-8B-Instruct | 33.7626 | 18.0452 | **21.5844** | 12.1687 | 73.2917 | 72.8501 | `constrast_infer_char_level_v2_20260825` |
| Qwen3-8B | 22.9506 | 10.7466 | 10.5834 | 5.4773 | 70.7644 | 70.5844 | `constrast_infer_char_level_v2_20260825` |
| Qwen3-8B (no thinking) | 25.8032 | 12.6379 | 13.2874 | 6.6139 | 72.4098 | 71.0900 | `constrast_infer_char_level_v2_20260825` |
| DeepSeek-R1-Distill-Qwen-7B | 22.2424 | 10.1437 | 11.1276 | 5.2028 | 70.4919 | 70.8488 | `constrast_infer_local_baseline_aligned_v1` |
| Qwen3.7-Plus | 28.9557 | 14.2997 | 13.9806 | 7.0564 | 72.4671 | 71.1095 | `constrast_infer_local_baseline_aligned_v1` |
| DeepSeek-V4-Pro-0813 | 31.5829 | 15.9347 | 16.5173 | 7.5108 | 72.8954 | 71.7077 | `constrast_infer_local_baseline_aligned_v1` |
| GLM-5 | 29.1407 | 14.3785 | 14.2190 | 7.4530 | 73.0394 | 71.8134 | `constrast_infer_local_baseline_aligned_v1` |

Metrics and the protocol recorded for each run are available in [`results/base_inference/direct_inference_metrics.json`](results/base_inference/direct_inference_metrics.json). Because historical runs record different metric protocol identifiers, compare them with the per-run protocol metadata rather than assuming all values came from an identical evaluation implementation.

<!--
## Model release

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuning method | PEFT LoRA |
| Training samples | 7,236 |
| Learning rate | `3e-4` |
| LoRA rank / alpha | `64 / 64` |
| LoRA dropout | `0` |
| Warmup steps | `50` |
| Epochs | `2` |
| Effective batch size | `2` |
| Maximum sequence length | `2048` |
| Scheduler | `linear` |
| Seed | `42` |
| Selected checkpoint | `checkpoint-7200` |
| Selection metric | Segmented BERTScore F1 |
| Recorded best metric | `79.76306056835053` |

Download the final adapter from [GitHub Releases](https://github.com/1wangjy/railway-llm-rag/releases/tag/v1.0.1). Verify the archive with the accompanying SHA-256 checksum before use. See [`model/MODEL_CARD.md`](model/MODEL_CARD.md) and [`model/RELEASE_MANIFEST.md`](model/RELEASE_MANIFEST.md) for metadata and integrity information.
-->

## Repository layout

```text
railway-llm-rag/
├── assets/          # README figures and dataset-composition tables
├── base_inference/  # local and Bailian-compatible direct inference
├── data/            # public data policy; no raw datasets
├── finetune/        # final LoRA configuration, launcher and evaluation
├── model/           # adapter metadata and release manifest
├── rag/             # Dify RAG batch evaluation client
└── results/         # metric-only summaries and checkpoint metadata
```

## Environment

Create the environment from `environment.yml` or install `requirements.txt`. Activate an environment containing PyTorch, Unsloth, PEFT, Transformers, and BERTScore before running the scripts.

```bash
conda env create -f environment.yml
conda activate unsloth
```

Alternatively:

```bash
python3 -m pip install -r requirements.txt
```

Set local paths for private assets; none of these assets are included in the repository:

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

The public configuration reproduces the selected LoRA setup. Review and adapt local paths, device placement, and memory-related options before training.

## Dify RAG evaluation

```bash
export DIFY_API_KEY='your-key'
export DIFY_BASE_URL='http://your-dify-host:port/v1'
export RAG_DATA_DIR=/path/to/private-data
export RAG_OUTPUT_DIR=outputs/rag

bash rag/run_batch_dify_rag_eval_unified.sh \
  --test-file "$EVAL_DATASET_PATH" \
  --run-name example
```

Do not commit API keys, Dify application credentials, private knowledge-base identifiers, raw responses, or trace logs.

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

## Data privacy and safety statement

This project concerns large language models for railway information-system operation and maintenance. Relevant materials may include railway O&M regulations, operating procedures, equipment and information-system documents, emergency plans, and expert knowledge. Users must ensure that all data used with this project has been lawfully obtained and appropriately authorized for research, development, and evaluation.

This repository does not distribute original railway documents, raw instruction data, internal system records, operational logs, credentials, network information, personal information, or other confidential materials. Only desensitized code, aggregate dataset statistics, selected evaluation results, and model-related metadata are released. Before using any data, users should remove personal identifiers, account information, access credentials, network addresses, equipment identifiers, proprietary technical details, and other information that could create security or privacy risks.

This project is intended for research and evaluation only. Model outputs may be incomplete, inaccurate, outdated, or misleading, and must not be treated as authoritative instructions for railway operation, equipment maintenance, dispatching, emergency response, cybersecurity, or other safety-critical activities. Any practical use must be reviewed and verified by qualified railway professionals against approved procedures and authoritative source documents.

Users are responsible for local data protection, access control, credential management, deployment security, and compliance with applicable laws, regulations, organizational policies, and information-security requirements. The authors do not guarantee suitability for any particular railway system or operational scenario and are not responsible for losses or risks caused by unauthorized data use, unverified model outputs, or improper deployment.

## Licence

No repository-wide licence has yet been selected. Contact the repository owner before reusing this code or any derivative artifact. The base model, LoRA adapter, source data, and third-party dependencies remain subject to their respective licences and acceptable-use requirements.
