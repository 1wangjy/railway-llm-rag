# Railway LLM RAG

Code and selected reproducibility artifacts for railway-domain question answering:
- LoRA fine-tuning of Qwen2.5-7B-Instruct;
- Dify-based RAG evaluation with operation-level trace diagnostics;
- direct inference of base and API models.

## Repository scope

This public repository intentionally excludes training/test data, railway knowledge-base source text, API keys, raw RAG traces, runtime logs, intermediate checkpoints, and model binaries.

The selected LoRA adapter is distributed separately as a GitHub Release. Download the release asset, verify its SHA-256 checksum, and load it on top of Qwen/Qwen2.5-7B-Instruct.

## Layout

- `finetune/`: training and generation-length evaluation scripts/configuration.
- `rag/`: Dify RAG batch evaluation and trace-diagnostic tooling.
- `base_inference/`: local and Bailian-compatible direct inference.
- `results/`: selected lightweight result summaries and final checkpoint metadata.
- `model/`: model-card and release metadata.
- `data/`: expected data policy and schema notes.

## Secrets

Set credentials only via environment variables or an untracked local `.env` file:

```bash
export DIFY_API_KEY='...'
export DASHSCOPE_API_KEY='...'
```

Never commit credentials.

## Reproducibility notes

Results depend on the unshared dataset version, Dify workflow publication version, knowledge-base configuration, retrieval parameters, and external model availability. The supplied files make code paths and recorded experiment settings auditable; they do not redistribute the underlying railway documents.
