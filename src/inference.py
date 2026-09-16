"""Model loading and inference."""

from pathlib import Path

try:
    import gradio as gr
except ImportError:
    gr = None

from src.project import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Model discovery (canonical location — import from here in scripts)
# ---------------------------------------------------------------------------

def find_available_models(models_dir: Path | None = None) -> list[Path]:
    """Find all model files, sorted TensorRT-first."""
    models_dir = models_dir or MODELS_DIR
    models = []
    if models_dir.exists():
        for ext in ("*.engine", "*.pt", "*.onnx"):
            models.extend(models_dir.glob(ext))
    return sorted(models, key=lambda p: p.suffix != ".engine")


def find_best_model(models_dir: Path | None = None) -> Path | None:
    """Return the best available model (TensorRT > PyTorch > ONNX).

    This is the single canonical implementation. Scripts should import this
    instead of duplicating the logic.
    """
    models = find_available_models(models_dir)
    return models[0] if models else None


def get_model_info(models_dir: Path | None = None) -> str:
    """Human-readable summary of available models."""
    models = find_available_models(models_dir)
    if not models:
        return "No models found in models/ folder."

    format_names = {".engine": "TensorRT", ".pt": "PyTorch", ".onnx": "ONNX"}
    lines = ["**Available models:**\n"]
    for m in models:
        size_mb = m.stat().st_size / (1024 * 1024)
        fmt = format_names.get(m.suffix, m.suffix)
        lines.append(f"- `{m.name}` ({fmt}, {size_mb:.1f} MB)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str | None = None):
    """Load a YOLO model from path or auto-discover the best one."""
    from ultralytics import YOLO

    if model_path and Path(model_path).exists():
        return YOLO(model_path)

    best = find_best_model()
    if best:
        return YOLO(str(best))

    raise FileNotFoundError("No model found. Place a .pt/.onnx/.engine file in models/.")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference_image(image, model_choice, confidence, iou_threshold):
    """Run inference on a single image."""
    if image is None:
        return None, "Please upload an image."

    try:
        if not 0 < confidence <= 1:
            return None, "Confidence must be between 0 and 1."
        if not 0 < iou_threshold <= 1:
            return None, "IoU threshold must be between 0 and 1."

        model_path = MODELS_DIR / model_choice if model_choice else None
        if model_path and not model_path.exists():
            return None, f"Model not found: {model_choice}"

        model = load_model(str(model_path) if model_path else None)
        results = model(image, conf=confidence, iou=iou_threshold, verbose=False)
        annotated = results[0].plot()

        num_detections = len(results[0].boxes) if hasattr(results[0], "boxes") else 0
        if num_detections > 0:
            confs = results[0].boxes.conf.cpu().numpy()
            stats = f"**Detections:** {num_detections}\n\n"
            stats += f"**Confidence:** {confs.min():.2f} - {confs.max():.2f}\n\n"
            stats += f"**Mean:** {confs.mean():.2f}"
        else:
            stats = "No detections found. Try lowering the confidence threshold."

        return annotated, stats

    except FileNotFoundError as e:
        return None, f"Model not found: {e}"
    except Exception as e:
        return None, f"Error: {e}"


def run_inference_video(video_path, model_choice, confidence, iou_threshold, progress=gr.Progress() if gr else None):
    """Run inference on a video file."""
    import cv2
    import tempfile

    if video_path is None:
        return None, "Please upload a video."

    try:
        model_path = MODELS_DIR / model_choice if model_choice else None
        if model_path and not model_path.exists():
            return None, f"Model not found: {model_choice}"

        model = load_model(str(model_path) if model_path else None)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Could not open video file."

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return None, "Video appears to be empty."

        output_path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_detections = 0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, iou=iou_threshold, verbose=False)
            annotated = results[0].plot()
            out.write(annotated)

            total_detections += len(results[0].boxes) if hasattr(results[0], "boxes") else 0
            frame_count += 1

            if progress:
                progress(frame_count / total_frames, desc=f"Frame {frame_count}/{total_frames}")

        cap.release()
        out.release()

        stats = f"**Processing complete!**\n\n"
        stats += f"**Frames:** {frame_count}\n\n"
        stats += f"**Total detections:** {total_detections}\n\n"
        if frame_count > 0:
            stats += f"**Avg per frame:** {total_detections / frame_count:.1f}"

        return output_path, stats

    except FileNotFoundError as e:
        return None, f"Model not found: {e}"
    except Exception as e:
        return None, f"Error: {e}"
