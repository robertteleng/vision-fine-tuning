#!/usr/bin/env python3
"""
Evaluation script — navigation obstacle detector

Compute mAP, precision, recall, and F1 on the validation set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model models/best.pt
    python scripts/evaluate.py --data data/dataset.yaml --conf 0.5
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import find_best_model
from src.project import find_dataset_yamls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate(args):
    from ultralytics import YOLO

    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Model not found: {args.model}")
            sys.exit(1)
    else:
        model_path = find_best_model()
        if not model_path:
            runs_dir = PROJECT_ROOT / "runs" / "train"
            if runs_dir.exists():
                bests = sorted(runs_dir.glob("*/weights/best.pt"), reverse=True)
                if bests:
                    model_path = bests[0]
            if not model_path:
                logger.error("No model found. Train one first or use --model.")
                sys.exit(1)

    logger.info(f"Model: {model_path}")
    model = YOLO(str(model_path))

    # Find dataset YAML
    if args.data:
        data_path = Path(args.data)
    else:
        yamls = find_dataset_yamls()
        data_path = yamls[0] if yamls else None

    if not data_path or not data_path.exists():
        logger.error(f"Dataset config not found. Use --data to specify one.")
        sys.exit(1)

    logger.info(f"Dataset: {data_path}")
    logger.info("=" * 60)
    logger.info("EVALUATING")
    logger.info("=" * 60)

    results = model.val(
        data=str(data_path),
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        save_json=args.save_json,
        verbose=True,
    )

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    metrics = {
        "Precision": results.box.mp,
        "Recall": results.box.mr,
        "mAP@50": results.box.map50,
        "mAP@50-95": results.box.map,
    }

    for name, value in metrics.items():
        logger.info(f"{name:15}: {value:.4f} ({value * 100:.1f}%)")

    if results.box.mp > 0 and results.box.mr > 0:
        f1 = 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr)
        logger.info(f"{'F1 Score':15}: {f1:.4f} ({f1 * 100:.1f}%)")

    logger.info("=" * 60)

    if hasattr(results.box, "ap_class_index") and len(results.box.ap_class_index) > 1:
        logger.info("\nPer-class metrics:")
        for i, cls_idx in enumerate(results.box.ap_class_index):
            cls_name = model.names[int(cls_idx)]
            logger.info(f"  {cls_name}: mAP50={results.box.ap50[i]:.3f}, mAP50-95={results.box.ap[i]:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Navigation detector — Evaluate")
    parser.add_argument("--model", "-m", default=None, help="Model .pt path")
    parser.add_argument("--data", "-d", default=None, help="Dataset YAML path")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (default: 0.001)")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (0.7, as in training validation)")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split to evaluate")
    parser.add_argument("--save-json", action="store_true", help="Save COCO JSON results")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
