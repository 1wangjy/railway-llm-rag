# Selected Railway Qwen2.5-7B LoRA Adapter

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Fine-tuning type: PEFT LoRA
- Selected adapter configuration: rank 64, alpha 64
- Learning rate: `3e-4`
- LoRA dropout: `0`
- Warmup steps: `50`
- Epochs: `2`
- Effective batch size: `2`
- Maximum sequence length: `2048`
- Scheduler: `linear`
- Seed: `42`
- Selection checkpoint: global step 7200
- Selection metric: segmented BERTScore F1
- Recorded best metric: 79.76306056835053

The release contains an adapter only, not the base model or dataset. Use is subject to the licences and acceptable-use conditions of the base model and all source data.

The label “selected final” indicates the model chosen by the experiment record. It does not imply that it is necessarily the highest score under every independent experimental condition.
