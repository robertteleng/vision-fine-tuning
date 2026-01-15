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

# Add project root to sys.path to allow imports from src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from src.inference import (
    find_available_models,
    get_model_info,
    run_inference_image,
    run_inference_video
)
from src.training import (
    start_training,
    get_training_status,
    get_training_metrics
)
from src.dataset import (
    get_dataset_stats,
    validate_dataset,
    export_dataset,
    DATA_DIR
)
from src.benchmark import run_benchmark
from src.annotation import (
    AutoAnnotator,
    AnnotationReviewer
)

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_TITLE = "VR Pillar Detector"
APP_DESCRIPTION = """
Detection of yellow/black signaling pillars in Virtual Reality environments.

**Model:** YOLOv12s fine-tuned | **Precision:** 98.7% mAP@50 | **Speed:** 180 FPS (TensorRT)
"""

# Global instances
auto_annotator = AutoAnnotator()
annotation_reviewer = AnnotationReviewer()

# ============================================================================
# WRAPPERS (To bridge Gradio and Classes)
# ============================================================================

def load_grounding_dino():
    """Load Grounding DINO model."""
    try:
        return auto_annotator.load_model()
    except Exception as e:
        return f"Error loading model: {str(e)}"


def annotate_single_image(image, prompt, threshold):
    """Wrapper for single image annotation."""
    return auto_annotator.annotate_image(image, prompt, threshold)


def annotate_folder_batch(source, output, prompt, threshold, val_split, copy_images, progress=gr.Progress()):
    """Wrapper for folder annotation."""
    return auto_annotator.annotate_folder(source, output, prompt, threshold, val_split, copy_images, progress)


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

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    """

    with gr.Blocks(title=APP_TITLE, css=custom_css, theme=gr.themes.Soft()) as app:

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
            # TAB: AUTO-ANNOTATION
            # ----------------------------------------------------------------
            with gr.TabItem("Auto-Annotate", id="auto_annotate"):
                gr.Markdown("## Zero-Shot Auto-Annotation with Grounding DINO")
                gr.Markdown("Automatically annotate images using natural language descriptions.")

                with gr.Row():
                    with gr.Column():
                        aa_model_status = gr.Markdown("**Model:** Not loaded")
                        aa_load_btn = gr.Button("Load Grounding DINO", variant="primary")

                with gr.Tabs():
                    # Single image annotation
                    with gr.TabItem("Single Image"):
                        with gr.Row():
                            with gr.Column():
                                aa_image = gr.Image(label="Input Image", type="numpy")
                                aa_prompt = gr.Textbox(
                                    label="Detection Prompt",
                                    value="yellow and black checkered pillar",
                                    placeholder="Describe the object to detect..."
                                )
                                aa_threshold = gr.Slider(
                                    0.1, 0.9, value=0.3, step=0.05,
                                    label="Confidence Threshold"
                                )
                                aa_detect_btn = gr.Button("Detect", variant="secondary")

                            with gr.Column():
                                aa_output = gr.Image(label="Result")
                                aa_stats = gr.Markdown()
                                aa_yolo = gr.Textbox(label="YOLO Format", lines=5)

                        aa_load_btn.click(load_grounding_dino, outputs=[aa_model_status])
                        aa_detect_btn.click(
                            annotate_single_image,
                            inputs=[aa_image, aa_prompt, aa_threshold],
                            outputs=[aa_output, aa_stats, aa_yolo]
                        )

                    # Batch annotation
                    with gr.TabItem("Batch Annotation"):
                        gr.Markdown("Annotate an entire folder and create a YOLO dataset.")

                        with gr.Row():
                            with gr.Column():
                                aa_source = gr.Textbox(
                                    label="Source Folder",
                                    placeholder="/path/to/images",
                                    value=str(DATA_DIR / "video_frames")
                                )
                                aa_output_folder = gr.Textbox(
                                    label="Output Folder",
                                    placeholder="/path/to/output",
                                    value=str(DATA_DIR / "dataset_auto")
                                )
                                aa_batch_prompt = gr.Textbox(
                                    label="Detection Prompt",
                                    value="yellow and black checkered pillar"
                                )
                                aa_batch_threshold = gr.Slider(
                                    0.1, 0.9, value=0.3, step=0.05,
                                    label="Confidence Threshold"
                                )
                                aa_val_split = gr.Slider(
                                    0.1, 0.4, value=0.2, step=0.05,
                                    label="Validation Split"
                                )
                                aa_copy = gr.Checkbox(label="Copy images (instead of symlinks)", value=False)
                                aa_batch_btn = gr.Button("Start Batch Annotation", variant="primary")

                            with gr.Column():
                                aa_batch_result = gr.Markdown("Click 'Start Batch Annotation' to begin.")

                        aa_batch_btn.click(
                            annotate_folder_batch,
                            inputs=[aa_source, aa_output_folder, aa_batch_prompt,
                                   aa_batch_threshold, aa_val_split, aa_copy],
                            outputs=[aa_batch_result]
                        )

            # ----------------------------------------------------------------
            # TAB: DATASET MANAGER
            # ----------------------------------------------------------------
            with gr.TabItem("Dataset", id="dataset"):
                gr.Markdown("## Dataset Manager")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Statistics")
                        ds_stats = gr.Markdown(get_dataset_stats())
                        ds_refresh_btn = gr.Button("Refresh Statistics")

                    with gr.Column():
                        gr.Markdown("### Validation")
                        ds_validation = gr.Markdown()
                        ds_validate_btn = gr.Button("Validate Dataset", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### Export Dataset")

                with gr.Row():
                    ds_export_format = gr.Radio(
                        choices=["ZIP (YOLO format)", "Copy to folder"],
                        value="ZIP (YOLO format)",
                        label="Export Format"
                    )
                    ds_export_name = gr.Textbox(
                        label="Output Name (optional)",
                        placeholder="Leave empty for auto-generated name"
                    )
                    ds_export_btn = gr.Button("Export", variant="primary")

                ds_export_result = gr.Markdown()

                ds_refresh_btn.click(get_dataset_stats, outputs=[ds_stats])
                ds_validate_btn.click(validate_dataset, outputs=[ds_validation])
                ds_export_btn.click(
                    export_dataset,
                    inputs=[ds_export_format, ds_export_name],
                    outputs=[ds_export_result]
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
