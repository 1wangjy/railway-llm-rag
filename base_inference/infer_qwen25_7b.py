"""Fixed Qwen2.5 inference entry point."""

import os

from configs.qwen25_7b_config import CONFIG

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
    "INFER_CUDA_VISIBLE_DEVICES",
    CONFIG.default_cuda_visible_devices,
)

from infer_common import main


if __name__ == "__main__":
    main(CONFIG)
