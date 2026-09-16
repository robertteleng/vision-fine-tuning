#!/usr/bin/env python3
"""
Export script — navigation obstacle detector

Export YOLO models to TensorRT, ONNX, and other formats.

Usage:
    python scripts/export_tensorrt.py --format engine --half
    python scripts/export_tensorrt.py --format onnx
    python scripts/export_tensorrt.py --model models/best.pt --format engine
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import find_best_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def export_model(args):
    """Export model to the specified format."""
    from ultralytics import YOLO
    import torch

    if args.format == "engine" and not torch.cuda.is_available():
        logger.error("TensorRT requires an NVIDIA GPU with CUDA.")
        sys.exit(1)

    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            # Check models/ directory
            alt_path = PROJECT_ROOT / "models" / args.model
            if alt_path.exists():
                model_path = alt_path
            else:
                logger.error(f"Model not found: {args.model}")
                sys.exit(1)
    else:
        model_path = find_best_model()
        if not model_path:
            # Search all runs subdirectories for best.pt
            runs_dir = PROJECT_ROOT / "runs"
            if runs_dir.exists():
                bests = sorted(runs_dir.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if bests:
                    model_path = bests[0]
            if not model_path:
                logger.error("No model found. Train one first or use --model.")
                sys.exit(1)

    logger.info(f"Model: {model_path}")
    model = YOLO(str(model_path))

    format_info = {
        "engine": "TensorRT (NVIDIA GPU optimized)",
        "onnx": "ONNX (portable)",
        "torchscript": "TorchScript",
        "openvino": "OpenVINO (Intel)",
        "coreml": "CoreML (Apple)",
        "tflite": "TensorFlow Lite (mobile)",
    }

    logger.info("=" * 60)
    logger.info(f"EXPORTING TO: {format_info.get(args.format, args.format)}")
    logger.info("=" * 60)

    export_args = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "dynamic": args.dynamic,
        "simplify": True,
        "verbose": True,
    }

    if args.format == "engine":
        export_args["device"] = 0
        if args.workspace:
            export_args["workspace"] = args.workspace
        logger.info(f"  Precision: {'FP16' if args.half else 'FP32'}")
        logger.info(f"  Image size: {args.imgsz}")

    try:
        export_path = model.export(**export_args)
        logger.info("\n" + "=" * 60)
        logger.info("EXPORT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Exported to: {export_path}")

        export_file = Path(export_path)
        if export_file.exists():
            size_mb = export_file.stat().st_size / (1024 * 1024)
            logger.info(f"Size: {size_mb:.1f} MB")

        logger.info(f"\nUsage:")
        logger.info(f"  python scripts/inference.py --source image.jpg --model {export_path}")
        logger.info(f"  python scripts/benchmark.py --model {model_path}")

        return export_path

    except Exception as e:
        logger.error(f"Export error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Navigation detector — Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/export_tensorrt.py --format engine --half    # TensorRT FP16
  python scripts/export_tensorrt.py --format onnx             # ONNX
  python scripts/export_tensorrt.py --model models/best.pt    # Specific model
""",
    )

    parser.add_argument("--model", "-m", default=None, help="Model .pt path")
    parser.add_argument("--format", "-f", default="engine",
                        choices=["engine", "onnx", "torchscript", "openvino", "coreml", "tflite"],
                        help="Export format (default: engine/TensorRT)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--half", action="store_true", help="FP16 precision")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT workspace GB")
    args = parser.parse_args()
    export_model(args)


if __name__ == "__main__":
    main()
