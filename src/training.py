from pathlib import Path
try:
    import gradio as gr
except ImportError:
    gr = None

PROJECT_ROOT = Path(__file__).parent.parent

def start_training(epochs, batch_size, model_base, progress=gr.Progress() if gr else None):
    """Start model training."""
    try:
        from ultralytics import YOLO
        import yaml
        import torch

        # Check GPU availability
        if not torch.cuda.is_available():
            return "⚠️ **Warning:** No GPU detected. Training will be slow on CPU.\n\nContinue anyway by clicking 'Start Training' again."

        # Load configuration
        config_path = PROJECT_ROOT / "config.yaml"
        if not config_path.exists():
            return f"❌ Configuration not found: {config_path}"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Dataset validation
        data_yaml = PROJECT_ROOT / "data" / "pillar.yaml"
        if not data_yaml.exists():
            return f"❌ Dataset config not found: {data_yaml}"

        # Validate dataset exists
        with open(data_yaml) as f:
            data_config = yaml.safe_load(f)

        dataset_path = Path(data_config.get('path', ''))
        if not dataset_path.exists():
            return f"❌ Dataset not found: {dataset_path}\n\nPlease check your pillar.yaml configuration."

        # Load base model
        model = YOLO(model_base)

        # Train
        # Note: We assume that the user wants to see the progress. 
        # Ultralytics doesn't easily plug into Gradio progress bar without custom callbacks,
        # but the verbose output will be printed to console.
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(PROJECT_ROOT / "runs" / "train"),
            verbose=True
        )

        return f"✅ **Training completed!**\n\nResults saved to: `{results.save_dir}`"

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            return f"❌ **Out of GPU memory!**\n\nTry reducing batch size to {batch_size // 2}."
        return f"❌ Runtime error: {str(e)}"
    except Exception as e:
        return f"❌ Error during training: {str(e)}"


def get_training_status():
    """Get training status."""
    runs_dir = PROJECT_ROOT / "runs" / "train"
    if not runs_dir.exists():
        return "No previous trainings"

    experiments = sorted(runs_dir.glob("*/"), reverse=True)
    if not experiments:
        return "No previous trainings"

    status = "## Previous Trainings\n\n"
    for exp in experiments[:5]:
        status += f"- `{exp.name}`\n"

        # Find metrics
        results_csv = exp / "results.csv"
        if results_csv.exists():
            import csv
            with open(results_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last = rows[-1]
                    status += f"  - Epochs: {len(rows)}\n"

    return status

def get_training_metrics():
    """Read metrics from last training (Hardcoded for demo purposes as in original app.py)."""
    # Ideally this should read from the latest run, but preserving original logic for now
    metrics = """
## Final Model Results (Training #4)

| Metric | Value |
|--------|-------|
| **mAP@50** | 98.7% |
| **mAP@50-95** | 87.4% |
| **Precision** | 91.7% |
| **Recall** | 99.2% |

## Inference Benchmark (RTX 2060)

| Format | Speed | FPS | Speedup |
|--------|-------|-----|---------|
| PyTorch (.pt) | 14.0 ms | 71 | 1x |
| ONNX (.onnx) | 19.4 ms | 52 | 0.7x |
| **TensorRT FP16** | **5.5 ms** | **180** | **2.5x** |

## Dataset

- **Training:** 621 images
- **Validation:** 139 images
- **Classes:** 1 (pillar)
"""
    return metrics
