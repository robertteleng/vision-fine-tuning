#!/usr/bin/env python3
"""
Export — step 5 of the pipeline.

Build a TensorRT engine on this machine (engines are not portable between
GPUs or TensorRT versions). INT8 calibrates on a sample of TRAIN images.
The benchmark builds engines itself; use this to get one for deployment.

Usage:
    uv sync --extra export
    uv run python scripts/export_tensorrt.py --weights models/yolo26n_nav.pt --precision fp16
    uv run python scripts/export_tensorrt.py --weights models/yolo26n_nav.pt --precision int8 --calib-images 1000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Navigation detector — TensorRT export")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/nav_combined/dataset.yaml",
                        help="Dataset YAML (INT8 calibration samples its train split)")
    parser.add_argument("--calib-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engines-dir", type=Path, default=PROJECT_ROOT / "models/engines")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        logger.error("TensorRT export needs an NVIDIA GPU with CUDA.")
        sys.exit(1)

    from src.trt_export import export_engine

    info = export_engine(args.weights, args.precision, args.data, args.engines_dir,
                         args.calib_images, args.seed, rebuild=args.rebuild)
    logger.info(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
