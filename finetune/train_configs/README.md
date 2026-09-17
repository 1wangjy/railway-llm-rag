# 训练配置分组规范

每次只比较一个训练参数，每个参数使用一个独立目录。目录内保存该参数不同取值的配置文件，训练日志自动进入同目录的 `logs/`。

所有训练配置均为完整独立字典，并统一设置 `bootstrap_enabled=False`。旧铁路 LoRA 仅作为独立对照模型，不作为参数搜索的初始化权重。

推荐结构：

```text
train_configs/
├── baseline/
│   ├── qwen25_7b_data7236_from_base.py
│   └── logs/
├── learning_rate/
│   ├── lr_1e-5.py
│   ├── lr_3e-5.py
│   ├── lr_5e-5.py
│   └── logs/
└── lora_r/
    ├── r_16.py
    ├── r_32.py
    ├── r_64.py
    └── logs/
```

新建实验组时，复制 baseline 配置，并确保每个配置使用不同的 `experiment_name`：

```bash
mkdir -p train_configs/learning_rate
cp train_configs/baseline/qwen25_7b_data7236_from_base.py train_configs/learning_rate/lr_1e-5.py
```

启动指定配置：

```bash
bash run_train.sh train_configs/learning_rate/lr_1e-5.py
```

后台启动：

```bash
nohup bash run_train.sh train_configs/learning_rate/lr_1e-5.py >/dev/null 2>&1 &
```

日志自动保存到：

```text
train_configs/learning_rate/logs/时间戳__lr_1e-5.log
```

生成长度不属于训练参数。相关消融配置位于 `evaluation_configs/generation_lengths.py`。
