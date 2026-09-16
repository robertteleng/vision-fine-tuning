#!/usr/bin/env python3
"""
Training script — navigation obstacle detector

Fine-tune YOLO26 with the recipe in config.yaml.

Usage:
    uv run python scripts/train.py --model yolo26n.pt --data data/nav_combined/dataset.yaml
    uv run python scripts/train.py --model yolo26s.pt --data data/nav_combined/dataset.yaml
    uv run python scripts/train.py --epochs 50 --batch 16
"""

import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware import detect_gpu, get_hardware_summary
from src.project import find_dataset_yamls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Log GPU info and return whether CUDA is available."""
    gpu = detect_gpu()
    logger.info("=" * 60)
    logger.info("HARDWARE")
    logger.info("=" * 60)
    logger.info(get_hardware_summary(gpu))
    logger.info("=" * 60)
    return gpu.available


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    logger.info(f"Loading config: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_experiment_name(base_name: str = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}" if base_name else f"train_{timestamp}"


def train(args):
    has_gpu = check_gpu()
    if not has_gpu and not args.force_cpu:
        logger.error("No GPU available. Use --force-cpu to train on CPU.")
        sys.exit(1)

    config = load_config(args.config)

    # CLI overrides
    if args.model:
        config["model"] = args.model
        logger.info(f"Override: model = {args.model}")
    for key in ("epochs", "batch", "imgsz"):
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val
            logger.info(f"Override: {key} = {val}")

    # Dataset YAML
    if args.data:
        data_yaml_path = args.data
    else:
        # Try config, then auto-discover
        data_yaml_path = config.get("data", None)
        if not data_yaml_path:
            yamls = find_dataset_yamls()
            if yamls:
                data_yaml_path = str(yamls[0])
            else:
                logger.error("No dataset YAML found. Use --data to specify one.")
                sys.exit(1)

    data_yaml_path = str(Path(data_yaml_path).resolve())
    logger.info(f"Dataset: {data_yaml_path}")

    from ultralytics import YOLO

    model_name = config.get("model", "yolo26n.pt")
    # Check models/ directory first, then fall back to Ultralytics auto-download
    model_path = PROJECT_ROOT / "models" / model_name
    if model_path.exists():
        model_name = str(model_path)
    logger.info(f"Base model: {model_name}")
    model = YOLO(model_name)

    exp_name = args.name or config.get("name") or get_experiment_name()

    # Build training params from config
    training_keys = (
        "epochs", "patience", "batch", "imgsz", "workers", "cache",
        "lr0", "lrf", "momentum", "weight_decay", "optimizer",
        "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "close_mosaic", "cos_lr",
        "fliplr", "flipud", "degrees", "translate", "scale", "shear",
        "perspective", "hsv_h", "hsv_s", "hsv_v", "mosaic", "mixup", "copy_paste",
        "save_period", "save", "plots", "device", "amp", "deterministic", "seed",
        "val", "verbose",
    )
    training_params = {k: config[k] for k in training_keys if k in config}
    training_params["data"] = data_yaml_path
    training_params["project"] = str(PROJECT_ROOT / config.get("project", "runs/train"))
    training_params["name"] = exp_name

    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    for k in ("epochs", "batch", "imgsz", "lr0", "device", "amp"):
        if k in training_params:
            logger.info(f"  {k}: {training_params[k]}")

    try:
        results = model.train(**training_params)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)

        if results and hasattr(results, "results_dict"):
            for key, value in results.results_dict.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("CUDA OUT OF MEMORY — reduce batch or imgsz in config.yaml")
        raise

    # Save versioned copy
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    best_model_path = (
        PROJECT_ROOT / config.get("project", "runs/train") / exp_name / "weights" / "best.pt"
    )

    if best_model_path.exists():
        import shutil

        ts = datetime.now().strftime("%Y%m%d")
        base = Path(model_name).stem
        versioned_name = f"{base}_{ts}.pt"
        versioned_path = models_dir / versioned_name
        shutil.copy(best_model_path, versioned_path)
        logger.info(f"Model saved: {versioned_path}")

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("=" * 60)
    logger.info(f"Results: {PROJECT_ROOT / config.get('project', 'runs/train') / exp_name}")
    logger.info(f"Next: python scripts/inference.py --source path/to/image.jpg")


def main():
    parser = argparse.ArgumentParser(
        description="Navigation detector — Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", default=str(PROJECT_ROOT / "config.yaml"), help="Config YAML path")
    parser.add_argument("--model", "-m", default=None, help="Model override (yolo26n.pt or yolo26s.pt)")
    parser.add_argument("--data", "-d", default=None, help="Dataset YAML path")
    parser.add_argument("--epochs", "-e", type=int, default=None, help="Epochs override")
    parser.add_argument("--batch", "-b", type=int, default=None, help="Batch size override (-1 = auto)")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size override")
    parser.add_argument("--name", "-n", default=None, help="Experiment name")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    try:
        train(args)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted. Checkpoints saved in runs/train/.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    main()
