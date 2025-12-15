#!/usr/bin/env python3
"""
VR Pillar Detector - Gradio Application

Web interface for pillar detection in VR environments.

Usage:
    python app.py
    python app.py --share  # Share publicly
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_TITLE = "VR Pillar Detector"
APP_DESCRIPTION = """
Detection of yellow/black signaling pillars in Virtual Reality environments.

**Model:** YOLOv12s fine-tuned | **Precision:** 98.7% mAP@50 | **Speed:** 180 FPS (TensorRT)
"""

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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


# ============================================================================
# TAB: INFERENCE
# ============================================================================

def run_inference_image(image, model_choice, confidence, iou_threshold):
    """Run inference on an image."""
    if image is None:
        return None, "Please upload an image"

    try:
        # Load model
        model_path = MODELS_DIR / model_choice if model_choice else None
        model = load_model(str(model_path) if model_path else None)

        # Run inference
        results = model(image, conf=confidence, iou=iou_threshold, verbose=False)

        # Get annotated image
        annotated = results[0].plot()

        # Statistics
        num_detections = len(results[0].boxes)
        if num_detections > 0:
            confs = results[0].boxes.conf.cpu().numpy()
            stats = f"**Detections:** {num_detections}\n\n"
            stats += f"**Confidence:** {confs.min():.2f} - {confs.max():.2f}\n\n"
            stats += f"**Mean:** {confs.mean():.2f}"
        else:
            stats = "**No pillars detected**"

        return annotated, stats

    except Exception as e:
        return None, f"Error: {str(e)}"


def run_inference_video(video_path, model_choice, confidence, iou_threshold, progress=gr.Progress()):
    """Run inference on a video."""
    import cv2
    import tempfile

    if video_path is None:
        return None, "Please upload a video"

    try:
        model_path = MODELS_DIR / model_choice if model_choice else None
        model = load_model(str(model_path) if model_path else None)

        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
            progress(frame_count / total_frames, desc=f"Frame {frame_count}/{total_frames}")

        cap.release()
        out.release()

        stats = f"**Frames processed:** {frame_count}\n\n"
        stats += f"**Total detections:** {total_detections}\n\n"
        stats += f"**Average per frame:** {total_detections/frame_count:.1f}"

        return output_path, stats

    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# TAB: METRICS
# ============================================================================

def get_training_metrics():
    """Read metrics from last training."""
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


def run_benchmark(iterations, progress=gr.Progress()):
    """Run speed benchmark."""
    import time
    import torch

    models = find_available_models()
    if not models:
        return "No models found"

    results = "## Benchmark Results\n\n"
    results += f"**Iterations:** {iterations}\n\n"

    # Dummy image
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    for model_path in progress.tqdm(models, desc="Evaluating models"):
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))

            # Warmup
            for _ in range(5):
                model(dummy_img, verbose=False)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                model(dummy_img, verbose=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            mean_ms = np.mean(times) * 1000
            fps = 1000 / mean_ms

            format_name = {
                ".engine": "TensorRT",
                ".pt": "PyTorch",
                ".onnx": "ONNX"
            }.get(model_path.suffix, model_path.suffix)

            results += f"### {format_name} (`{model_path.name}`)\n"
            results += f"- **Speed:** {mean_ms:.2f} ms\n"
            results += f"- **FPS:** {fps:.1f}\n\n"

        except Exception as e:
            results += f"### {model_path.name}\n- Error: {str(e)}\n\n"

    return results


# ============================================================================
# TAB: TRAINING
# ============================================================================

def start_training(epochs, batch_size, model_base, progress=gr.Progress()):
    """Start model training."""
    try:
        from ultralytics import YOLO
        import yaml

        # Load configuration
        config_path = PROJECT_ROOT / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Update parameters
        config["epochs"] = epochs
        config["batch"] = batch_size

        # Load base model
        model = YOLO(model_base)

        # Dataset
        data_yaml = PROJECT_ROOT / "data" / "pillar.yaml"

        # Train
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(PROJECT_ROOT / "runs" / "train"),
            verbose=True
        )

        return f"Training completed. Results in: {results.save_dir}"

    except Exception as e:
        return f"Error during training: {str(e)}"


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


# ============================================================================
# TAB: ANNOTATIONS
# ============================================================================

class AnnotationReviewer:
    """Annotation reviewer for Gradio."""

    def __init__(self):
        self.images_dir = None
        self.labels_dir = None
        self.image_files = []
        self.current_idx = 0
        self.annotations = []

    def load_dataset(self, dataset_path: str):
        """Load a dataset for review."""
        dataset_path = Path(dataset_path) if dataset_path else DATA_DIR / "dataset" / "train"

        self.images_dir = dataset_path / "images"
        self.labels_dir = dataset_path / "labels"

        if not self.images_dir.exists():
            return None, f"Not found: {self.images_dir}", "0 / 0"

        self.image_files = sorted(self.images_dir.glob("*.jpg"))
        if not self.image_files:
            self.image_files = sorted(self.images_dir.glob("*.png"))

        if not self.image_files:
            return None, "No images found", "0 / 0"

        self.current_idx = 0
        return self._load_current()

    def _load_current(self):
        """Load current image and annotations."""
        if not self.image_files:
            return None, "No images", "0 / 0"

        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            return None, f"Error loading: {img_path.name}", f"{self.current_idx + 1} / {len(self.image_files)}"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Load annotations
        self.annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        self.annotations.append({
                            'id': i,
                            'cls': int(parts[0]),
                            'x': float(parts[1]),
                            'y': float(parts[2]),
                            'w': float(parts[3]),
                            'h': float(parts[4])
                        })

        # Draw annotations
        img_display = img.copy()
        for ann in self.annotations:
            cx, cy = int(ann['x'] * w), int(ann['y'] * h)
            bw, bh = int(ann['w'] * w), int(ann['h'] * h)
            x1, y1 = cx - bw // 2, cy - bh // 2
            x2, y2 = cx + bw // 2, cy + bh // 2

            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_display, str(ann['id']), (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        info = f"**{img_path.name}** - {len(self.annotations)} annotations"
        counter = f"{self.current_idx + 1} / {len(self.image_files)}"

        return img_display, info, counter

    def next_image(self):
        """Next image."""
        if self.image_files and self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
        return self._load_current()

    def prev_image(self):
        """Previous image."""
        if self.image_files and self.current_idx > 0:
            self.current_idx -= 1
        return self._load_current()

    def goto_image(self, idx: int):
        """Go to specific image."""
        if self.image_files and 0 <= idx - 1 < len(self.image_files):
            self.current_idx = idx - 1
        return self._load_current()

    def delete_annotation(self, ann_id: int):
        """Delete an annotation and save."""
        if not self.image_files:
            return self._load_current()

        # Filter annotation
        self.annotations = [a for a in self.annotations if a['id'] != ann_id]

        # Save
        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        with open(label_path, 'w') as f:
            for ann in self.annotations:
                f.write(f"{ann['cls']} {ann['x']:.6f} {ann['y']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n")

        return self._load_current()

    def delete_all_annotations(self):
        """Delete all annotations from current image."""
        if not self.image_files:
            return self._load_current()

        self.annotations = []

        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        with open(label_path, 'w') as f:
            pass  # Empty file

        return self._load_current()


# Global reviewer instance
annotation_reviewer = AnnotationReviewer()


def load_annotations_dataset(dataset_choice):
    """Wrapper to load dataset."""
    if dataset_choice == "Train":
        path = DATA_DIR / "dataset" / "train"
    elif dataset_choice == "Val":
        path = DATA_DIR / "dataset" / "val"
    else:  # video_what
        path = DATA_DIR / "dataset_what"
    return annotation_reviewer.load_dataset(str(path))


def annotations_next():
    return annotation_reviewer.next_image()


def annotations_prev():
    return annotation_reviewer.prev_image()


def annotations_goto(idx):
    return annotation_reviewer.goto_image(int(idx))


def annotations_delete(ann_id):
    return annotation_reviewer.delete_annotation(int(ann_id))


def annotations_delete_all():
    return annotation_reviewer.delete_all_annotations()


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_app():
    """Create the Gradio application."""

    # Model list for dropdown
    model_choices = [m.name for m in find_available_models()]
    default_model = model_choices[0] if model_choices else None

    with gr.Blocks(title=APP_TITLE) as app:

        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        with gr.Tabs():
            # ----------------------------------------------------------------
            # TAB: INFERENCE
            # ----------------------------------------------------------------
            with gr.TabItem("Inference", id="inference"):
                with gr.Tabs():
                    # Sub-tab: Image
                    with gr.TabItem("Image"):
                        with gr.Row():
                            with gr.Column():
                                img_input = gr.Image(
                                    label="Input image",
                                    type="numpy"
                                )
                                with gr.Row():
                                    img_model = gr.Dropdown(
                                        choices=model_choices,
                                        value=default_model,
                                        label="Model"
                                    )
                                with gr.Row():
                                    img_conf = gr.Slider(
                                        0.1, 1.0, value=0.65,
                                        step=0.05,
                                        label="Min confidence"
                                    )
                                    img_iou = gr.Slider(
                                        0.1, 1.0, value=0.45,
                                        step=0.05,
                                        label="IoU threshold"
                                    )
                                img_btn = gr.Button("Detect", variant="primary")

                            with gr.Column():
                                img_output = gr.Image(label="Result")
                                img_stats = gr.Markdown()

                        img_btn.click(
                            run_inference_image,
                            inputs=[img_input, img_model, img_conf, img_iou],
                            outputs=[img_output, img_stats]
                        )

                    # Sub-tab: Video
                    with gr.TabItem("Video"):
                        with gr.Row():
                            with gr.Column():
                                vid_input = gr.Video(label="Input video")
                                with gr.Row():
                                    vid_model = gr.Dropdown(
                                        choices=model_choices,
                                        value=default_model,
                                        label="Model"
                                    )
                                with gr.Row():
                                    vid_conf = gr.Slider(
                                        0.1, 1.0, value=0.65,
                                        step=0.05,
                                        label="Min confidence"
                                    )
                                    vid_iou = gr.Slider(
                                        0.1, 1.0, value=0.45,
                                        step=0.05,
                                        label="IoU threshold"
                                    )
                                vid_btn = gr.Button("Process Video", variant="primary")

                            with gr.Column():
                                vid_output = gr.Video(label="Result")
                                vid_stats = gr.Markdown()

                        vid_btn.click(
                            run_inference_video,
                            inputs=[vid_input, vid_model, vid_conf, vid_iou],
                            outputs=[vid_output, vid_stats]
                        )

            # ----------------------------------------------------------------
            # TAB: METRICS
            # ----------------------------------------------------------------
            with gr.TabItem("Metrics", id="metrics"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Model Metrics")
                        metrics_display = gr.Markdown(get_training_metrics())

                    with gr.Column():
                        gr.Markdown("## Available Models")
                        models_info = gr.Markdown(get_model_info())

                gr.Markdown("---")
                gr.Markdown("## Live Benchmark")

                with gr.Row():
                    bench_iterations = gr.Slider(
                        10, 200, value=50,
                        step=10,
                        label="Iterations"
                    )
                    bench_btn = gr.Button("Run Benchmark", variant="secondary")

                bench_results = gr.Markdown()

                bench_btn.click(
                    run_benchmark,
                    inputs=[bench_iterations],
                    outputs=[bench_results]
                )

            # ----------------------------------------------------------------
            # TAB: TRAINING
            # ----------------------------------------------------------------
            with gr.TabItem("Training", id="training"):
                gr.Markdown("## Training Configuration")

                with gr.Row():
                    with gr.Column():
                        train_epochs = gr.Slider(
                            10, 200, value=50,
                            step=10,
                            label="Epochs"
                        )
                        train_batch = gr.Slider(
                            4, 16, value=8,
                            step=2,
                            label="Batch Size"
                        )
                        train_model = gr.Dropdown(
                            choices=["yolo12s.pt", "yolo12n.pt", "yolo11s.pt", "yolo11n.pt"],
                            value="yolo12s.pt",
                            label="Base Model"
                        )
                        train_btn = gr.Button("Start Training", variant="primary")

                    with gr.Column():
                        gr.Markdown("## Status")
                        train_status = gr.Markdown(get_training_status())

                train_output = gr.Markdown()

                train_btn.click(
                    start_training,
                    inputs=[train_epochs, train_batch, train_model],
                    outputs=[train_output]
                )

            # ----------------------------------------------------------------
            # TAB: ANNOTATIONS
            # ----------------------------------------------------------------
            with gr.TabItem("Annotations", id="annotations"):
                gr.Markdown("## Annotation Reviewer")
                gr.Markdown("Review and edit dataset annotations.")

                with gr.Row():
                    ann_dataset = gr.Radio(
                        choices=["Train", "Val", "video_what"],
                        value="video_what",
                        label="Dataset"
                    )
                    ann_load_btn = gr.Button("Load Dataset", variant="primary")

                with gr.Row():
                    with gr.Column(scale=3):
                        ann_image = gr.Image(label="Image with annotations")

                    with gr.Column(scale=1):
                        ann_info = gr.Markdown("Select a dataset")
                        ann_counter = gr.Textbox(label="Progress", value="0 / 0", interactive=False)

                        with gr.Row():
                            ann_prev_btn = gr.Button("< Previous")
                            ann_next_btn = gr.Button("Next >")

                        ann_goto = gr.Number(label="Go to image #", value=1, precision=0)
                        ann_goto_btn = gr.Button("Go")

                        gr.Markdown("---")
                        gr.Markdown("### Delete annotations")
                        ann_delete_id = gr.Number(label="Annotation ID", value=0, precision=0)
                        ann_delete_btn = gr.Button("Delete annotation", variant="secondary")
                        ann_delete_all_btn = gr.Button("Delete ALL", variant="stop")

                # Event handlers
                ann_load_btn.click(
                    load_annotations_dataset,
                    inputs=[ann_dataset],
                    outputs=[ann_image, ann_info, ann_counter]
                )
                ann_next_btn.click(
                    annotations_next,
                    outputs=[ann_image, ann_info, ann_counter]
                )
                ann_prev_btn.click(
                    annotations_prev,
                    outputs=[ann_image, ann_info, ann_counter]
                )
                ann_goto_btn.click(
                    annotations_goto,
                    inputs=[ann_goto],
                    outputs=[ann_image, ann_info, ann_counter]
                )
                ann_delete_btn.click(
                    annotations_delete,
                    inputs=[ann_delete_id],
                    outputs=[ann_image, ann_info, ann_counter]
                )
                ann_delete_all_btn.click(
                    annotations_delete_all,
                    outputs=[ann_image, ann_info, ann_counter]
                )

            # ----------------------------------------------------------------
            # TAB: INFO
            # ----------------------------------------------------------------
            with gr.TabItem("Info", id="info"):
                gr.Markdown("""
## About the Project

**VR Pillar Detector** is an object detection model specialized in identifying
yellow/black signaling pillars in Virtual Reality environments.

### Features

- **Model:** YOLOv12s fine-tuned
- **Dataset:** 760 manually annotated images
- **Precision:** 98.7% mAP@50
- **Speed:** 180 FPS with TensorRT FP16

### Recommended Hardware

- NVIDIA GPU with CUDA (RTX 2060 or higher)
- 6GB+ VRAM for training
- 2GB+ VRAM for inference

### CLI Usage

```bash
# Image inference
python scripts/inference.py --source image.jpg

# Video inference
python scripts/inference.py --source video.mp4 --model models/best.engine

# Training
python scripts/train.py

# Benchmark
python scripts/benchmark.py
```

### Links

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [YOLOv12](https://docs.ultralytics.com/models/yolo12/)
                """)

        gr.Markdown("---")
        gr.Markdown("*Developed with Ultralytics YOLO and Gradio*")

    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="VR Pillar Detector - Web Interface")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
