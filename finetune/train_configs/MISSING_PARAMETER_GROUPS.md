# Single-factor parameter groups

The previously missing groups have been added:

- `num_train_epochs`: 2, 3
- `warmup`: fixed steps 5, 200, 300; ratio 0.03
- `lora_alpha`: 32, 64
- `lr_scheduler_type`: linear, cosine

Every Python configuration file is now self-contained and explicitly records
the complete training configuration. Files do not import or inherit a baseline
configuration. This makes a winning value safe to combine with winning values
from other parameter groups.

The ratio-based warmup configuration explicitly sets `warmup_steps=0`, because
fixed-step and ratio-based warmup are mutually exclusive. `lora_alpha` files
also explicitly record the complete LoRA structure, including `lora_r`.
