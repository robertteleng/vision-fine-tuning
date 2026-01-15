import cv2
import numpy as np
import tempfile
from pathlib import Path
try:
    import gradio as gr
except ImportError:
    gr = None

# We need to define PROJECT_ROOT or pass it in. 
# For now, we'll assume the caller might configure paths or we use relative paths.
# However, the original code used PROJECT_ROOT based on __file__.parent of app.py.
# In src/inference.py, __file__.parent is src/. So PROJECT_ROOT would be ../
# Let's define PROJECT_ROOT locally here relative to this file.

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

def find_available_models():
    """Find all available models."""
    models = []
    if MODELS_DIR.exists():
        for ext in ["*.engine", "*.pt", "*.onnx"]:
            models.extend(MODELS_DIR.glob(ext))
    return sorted(models, key=lambda x: x.suffix != ".engine")  # TensorRT first


def get_model_info():
    """Get information about available models."""
    models = find_available_models()
    if not models:
        return "No models found in models/ folder"

    info = "**Available models:**\n\n"
    for m in models:
        size_mb = m.stat().st_size / (1024 * 1024)
        format_name = {
            ".engine": "TensorRT FP16",
            ".pt": "PyTorch",
            ".onnx": "ONNX"
        }.get(m.suffix, m.suffix)
        info += f"- `{m.name}` ({format_name}, {size_mb:.1f} MB)\n"

    return info


def load_model(model_path: str = None):
    """Load the YOLO model."""
    from ultralytics import YOLO

    if model_path and Path(model_path).exists():
        return YOLO(model_path)

    # Find best available model
    models = find_available_models()
    if models:
        return YOLO(str(models[0]))

    raise FileNotFoundError("No model found")


def run_inference_image(image, model_choice, confidence, iou_threshold):
    """Run inference on an image."""
    if image is None:
        return None, "⚠️ Please upload an image"

    try:
        # Validate parameters
        if not 0 < confidence <= 1:
            return None, "⚠️ Confidence must be between 0 and 1"
        if not 0 < iou_threshold <= 1:
            return None, "⚠️ IoU threshold must be between 0 and 1"

        # Load model
        model_path = MODELS_DIR / model_choice if model_choice else None
        if model_path and not model_path.exists():
            return None, f"❌ Model not found: {model_choice}"

        model = load_model(str(model_path) if model_path else None)

        # Run inference
        results = model(image, conf=confidence, iou=iou_threshold, verbose=False)

        # Get annotated image
        annotated = results[0].plot()

        # Statistics
        num_detections = len(results[0].boxes)
        if num_detections > 0:
            confs = results[0].boxes.conf.cpu().numpy()
            stats = f"✅ **Detections:** {num_detections}\n\n"
            stats += f"**Confidence:** {confs.min():.2f} - {confs.max():.2f}\n\n"
            stats += f"**Mean:** {confs.mean():.2f}"
        else:
            stats = "ℹ️ **No pillars detected**\n\nTry lowering the confidence threshold."

        return annotated, stats

    except FileNotFoundError as e:
        return None, f"❌ Model not found: {str(e)}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def run_inference_video(video_path, model_choice, confidence, iou_threshold, progress=gr.Progress() if gr else None):
    """Run inference on a video."""
    
    if video_path is None:
        return None, "⚠️ Please upload a video"

    try:
        # Validate model
        model_path = MODELS_DIR / model_choice if model_choice else None
        if model_path and not model_path.exists():
            return None, f"❌ Model not found: {model_choice}"

        model = load_model(str(model_path) if model_path else None)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "❌ Could not open video file"

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return None, "❌ Video appears to be empty"

        # Create output video
        output_path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_detections = 0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            results = model(frame, conf=confidence, iou=iou_threshold, verbose=False)
            annotated = results[0].plot()
            out.write(annotated)

            total_detections += len(results[0].boxes)
            frame_count += 1

            # Update progress
            if progress:
                progress(frame_count / total_frames, desc=f"Frame {frame_count}/{total_frames}")

        cap.release()
        out.release()

        stats = f"✅ **Processing complete!**\n\n"
        stats += f"**Frames processed:** {frame_count}\n\n"
        stats += f"**Total detections:** {total_detections}\n\n"
        if frame_count > 0:
            stats += f"**Average per frame:** {total_detections/frame_count:.1f}"

        return output_path, stats

    except FileNotFoundError as e:
        return None, f"❌ Model not found: {str(e)}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"
