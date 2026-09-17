# 生成长度消融

`generation_lengths.py` 是参数配置，`../evaluate_generation_lengths.py` 是可执行评估器。二者只用于同一个最佳 LoRA adapter 的推理评估，不参与训练，也不改变 adapter。

所有长度必须使用同一个模型、同一份数据、`do_sample=False`，分别记录质量指标、达到长度上限的比例、平均生成长度和运行时间。

## 运行

```bash
cd /data16T/wjy/Learn_llm/铁路大模型/new_7b
./run_generation_length_ablation.sh \
  --adapter-path /absolute/path/to/best_adapter
```

默认使用物理 GPU 2。需要换卡时使用：

```bash
ABLATION_CUDA_VISIBLE_DEVICES=4 ./run_generation_length_ablation.sh \
  --adapter-path /absolute/path/to/best_adapter
```

可用参数：

- `--lengths 128 256`：只评估指定长度，适合冒烟测试。
- `--max-samples 2`：只读取前 N 条样本，适合检查流程。
- `--output-dir PATH`：覆盖默认输出目录。
- `--overwrite`：重新计算已经存在的长度结果；默认会复用已完成结果。

使用 `--max-samples` 冒烟测试时应同时指定单独的 `--output-dir`，避免和正式 799 条结果混在一起。

评估器会在启动时校验 `adapter_config.json`，并确认其中记录的基础模型与配置一致。每个长度产生一个 `max_new_tokens_<N>.json`，总表写入 `summary.json` 和 `summary.csv`。输出中同时保存运行配置和 adapter 路径，便于复现实验。

正式消融不要使用 `--max-samples`，也不要在不同长度之间更换 adapter、数据或生成参数。
