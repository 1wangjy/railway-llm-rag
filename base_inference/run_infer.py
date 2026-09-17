"""Unified entry point for all baseline inference jobs."""

import argparse
import os
import sys

from infer_config import get_config, model_configs


def select_model():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model",
        choices=sorted(model_configs),
        required=True,
        help="Model configuration key defined in infer_config.py.",
    )
    return parser.parse_known_args()


def main():
    selected, remaining = select_model()
    config = get_config(selected.model)

    # CUDA must be selected before importing Unsloth/Torch in infer_common.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
        "INFER_CUDA_VISIBLE_DEVICES",
        config.default_cuda_visible_devices,
    )

    # The shared implementation parses all remaining inference arguments.
    sys.argv = [sys.argv[0], *remaining]
    from infer_common import main as run_common_inference

    run_common_inference(config)


if __name__ == "__main__":
    main()
