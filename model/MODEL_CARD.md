# Selected Railway Qwen2.5-7B LoRA Adapter

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Fine-tuning type: PEFT LoRA
- Selected adapter configuration: rank 64, alpha 64
- Selection checkpoint: global step 6600
- Selection metric: segmented BERTScore F1
- Recorded best metric: 79.6344270352623

The release contains an adapter only, not the base model or dataset. Use is subject to the licences and acceptable-use conditions of the base model and all source data.

The label “selected final” indicates the model chosen by the experiment record. It does not imply that it is necessarily the highest score under every independent experimental condition.
