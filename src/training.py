"""Training routines for YOLO models."""

import csv
from pathlib import Path

try:
    import gradio as gr
except ImportError:
    gr = None

from src.hardware import detect_gpu, get_hardware_summary
from src.project import PROJECT_ROOT, find_dataset_yamls

RUNS_DIR = PROJECT_ROOT / "runs" / "train"


def start_training(epochs, batch_size, model_base, data_yaml_path, progress=gr.Progress() if gr else None):
    """Start model training.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size (-1 for auto).
        model_base: Base model name, e.g. "yolo26m.pt".
        data_yaml_path: Path to dataset YAML file.
    """
    try:
        from ultralytics import YOLO
        import yaml

        gpu = detect_gpu()
        if not gpu.available:
            return "Warning: No GPU detected. Training will run on CPU."

        # Dataset validation
        data_yaml = Path(data_yaml_path)
        if not data_yaml.exists():
            return f"Error: Dataset config not found: {data_yaml}"

        with open(data_yaml) as f:
            data_config = yaml.safe_load(f)

        dataset_path = Path(data_config.get("path", ""))
        if not dataset_path.exists():
            return f"Error: Dataset not found at: {dataset_path}"

        # Load config.yaml for additional training params
        config_path = PROJECT_ROOT / "config.yaml"
        extra_params = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            # Pick training-relevant keys only
            for key in ("imgsz", "patience", "lr0", "lrf", "momentum", "weight_decay",
                        "optimizer", "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
                        "fliplr", "flipud", "degrees", "translate", "scale", "shear",
                        "perspective", "hsv_h", "hsv_s", "hsv_v", "mosaic", "mixup",
                        "copy_paste", "amp", "save_period", "plots", "verbose"):
                if key in config:
                    extra_params[key] = config[key]

        model = YOLO(model_base)

        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            project=str(RUNS_DIR),
            **extra_params,
        )

        return f"Training completed!\n\nResults saved to: {results.save_dir}"

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            suggested = max(1, batch_size // 2) if batch_size > 0 else 4
            return f"Out of GPU memory. Try batch size {suggested}."
        return f"Runtime error: {e}"
    except Exception as e:
        return f"Error during training: {e}"


def get_training_status():
    """List recent training runs."""
    if not RUNS_DIR.exists():
        return "No previous trainings."

    experiments = sorted(RUNS_DIR.glob("*/"), reverse=True)
    if not experiments:
        return "No previous trainings."

    status = "## Previous Trainings\n\n"
    for exp in experiments[:5]:
        status += f"- `{exp.name}`\n"
        results_csv = exp / "results.csv"
        if results_csv.exists():
            with open(results_csv) as f:
                rows = list(csv.DictReader(f))
                if rows:
                    status += f"  - Epochs: {len(rows)}\n"

    return status


def get_training_metrics():
    """Read metrics from the latest training run dynamically."""
    if not RUNS_DIR.exists():
        return "No training runs found. Train a model first."

    experiments = sorted(RUNS_DIR.glob("*/"), reverse=True)
    if not experiments:
        return "No training runs found. Train a model first."

    # Find latest run with results.csv
    for exp in experiments:
        results_csv = exp / "results.csv"
        if not results_csv.exists():
            continue

        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        last = rows[-1]

        # Ultralytics CSV column names (strip whitespace)
        clean = {k.strip(): v.strip() for k, v in last.items()}

        # Extract metrics (column names vary slightly across versions)
        def get_metric(keys, fmt=".3f"):
            for k in keys:
                if k in clean:
                    try:
                        val = float(clean[k])
                        return f"{val:{fmt}}"
                    except ValueError:
                        pass
            return "N/A"

        precision = get_metric(["metrics/precision(B)", "metrics/precision"])
        recall = get_metric(["metrics/recall(B)", "metrics/recall"])
        map50 = get_metric(["metrics/mAP50(B)", "metrics/mAP50"])
        map50_95 = get_metric(["metrics/mAP50-95(B)", "metrics/mAP50-95"])

        gpu = detect_gpu()
        hw = get_hardware_summary(gpu)

        return f"""## Latest Training: `{exp.name}`

| Metric | Value |
|--------|-------|
| Precision | {precision} |
| Recall | {recall} |
| mAP@50 | {map50} |
| mAP@50-95 | {map50_95} |
| Epochs | {len(rows)} |

{hw}
"""

    return "No results.csv found in any training run."
