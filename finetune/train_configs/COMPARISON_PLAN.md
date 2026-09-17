# 单因素训练配置清单

所有配置都是完整、独立的配置文件。参数搜索统一从 Qwen2.5-7B-Instruct 基础模型和同一随机初始化 LoRA 开始，不加载旧 LoRA。

| 实验组 | 取值 | 配置数量 |
|---|---|---:|
| max_seq_length | 1024、1536、2048、3072、4096 | 5 |
| 有效 batch size | 1、2、4、8（单卡 batch 固定为 1） | 4 |
| lora_r | 8、16、32、64、128 | 5 |
| lora_alpha | 32、64 | 2 |
| lora_dropout | 0、0.05、0.1 | 3 |
| learning_rate | 1e-5、2e-5、3e-5、5e-5、7e-5、1e-4 | 6 |
| lr_scheduler_type | linear、cosine | 2 |
| num_train_epochs | 2、3 | 2 |
| warmup | steps 5、200、300，ratio 0.03 | 4 |

共34份分组配置。序列长度实验主要修改 training_max_length；1024配置仍以2048加载模型，以容纳固定1024-token生成评估。LoRA r 实验保持 lora_alpha=32；有效 batch 实验保持 per_device_train_batch_size=1，只修改 gradient_accumulation_steps。

`generation_max_new_tokens` 固定为1024，不再作为训练参数。128、256、512、1024、1536的比较移动到 `evaluation_configs/generation_lengths.py`，对同一个最佳 checkpoint 做推理消融。
