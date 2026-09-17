"""Ling-3.0-tiny pure-inference configuration.

The model uses custom remote-code architecture, so the shared inference
loader will fall back to Transformers when Unsloth does not support it.
"""

from copy import copy

from configs.qwen3_8b_config import CONFIG as _BASE_CONFIG


CONFIG = copy(_BASE_CONFIG)
CONFIG.key = "ling3_tiny"
CONFIG.description = "Evaluate Ling-3.0-tiny with reasoning disabled."
CONFIG.model_path = "/data16T/wjy/models/Ling-3.0-tiny"
CONFIG.default_cuda_visible_devices = "3"
CONFIG.output_path = "outputs/ling3_tiny_base_testdata799_system_max1024_no_thinking_predictions.json"
CONFIG.enable_thinking = False
CONFIG.trust_remote_code = True
