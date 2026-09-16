#!/usr/bin/env python3
"""
Navigation obstacle detector — Gradio demo

Run the trained detector on an image or a video and inspect the latest
training metrics.

Usage:
    uv run python app.py
    uv run python app.py --share  # temporary public link
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from src.inference import (
    find_available_models,
    get_model_info,
    run_inference_image,
    run_inference_video,
)
from src.training import get_training_metrics
from src.hardware import detect_gpu, get_hardware_summary

APP_TITLE = "Navigation Obstacle Detector"
APP_DESCRIPTION = """YOLO26 fine-tuned on 24 classes that matter to a blind or low-vision
pedestrian (people, vehicles, doors, stairs, street furniture). Built for AriaGuard."""


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_app():
    model_choices = [m.name for m in find_available_models()]
    default_model = model_choices[0] if model_choices else None

    gpu = detect_gpu()
    hw_summary = get_hardware_summary(gpu)

    custom_css = """
    .gradio-container { max-width: 1400px !important; }
    """

    with gr.Blocks(title=APP_TITLE, css=custom_css, theme=gr.themes.Soft()) as app:

        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)
        gr.Markdown(f"\n{hw_summary}")

        with gr.Tabs():
            # ----------------------------------------------------------------
            # TAB: INFERENCE
            # ----------------------------------------------------------------
            with gr.TabItem("Inference", id="inference"):
                with gr.Tabs():
                    with gr.TabItem("Image"):
                        with gr.Row():
                            with gr.Column():
                                img_input = gr.Image(label="Input image", type="numpy")
                                with gr.Row():
                                    img_model = gr.Dropdown(
                                        choices=model_choices, value=default_model, label="Model"
                                    )
                                with gr.Row():
                                    img_conf = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Confidence")
                                    img_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU threshold")
                                img_btn = gr.Button("Detect", variant="primary")

                            with gr.Column():
                                img_output = gr.Image(label="Result")
                                img_stats = gr.Markdown()

                        img_btn.click(
                            run_inference_image,
                            inputs=[img_input, img_model, img_conf, img_iou],
                            outputs=[img_output, img_stats],
                        )

                    with gr.TabItem("Video"):
                        with gr.Row():
                            with gr.Column():
                                vid_input = gr.Video(label="Input video")
                                with gr.Row():
                                    vid_model = gr.Dropdown(
                                        choices=model_choices, value=default_model, label="Model"
                                    )
                                with gr.Row():
                                    vid_conf = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Confidence")
                                    vid_iou = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="IoU threshold")
                                vid_btn = gr.Button("Process Video", variant="primary")

                            with gr.Column():
                                vid_output = gr.Video(label="Result")
                                vid_stats = gr.Markdown()

                        vid_btn.click(
                            run_inference_video,
                            inputs=[vid_input, vid_model, vid_conf, vid_iou],
                            outputs=[vid_output, vid_stats],
                        )

            # ----------------------------------------------------------------
            # TAB: METRICS
            # ----------------------------------------------------------------
            with gr.TabItem("Metrics", id="metrics"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Training Metrics")
                        metrics_display = gr.Markdown(get_training_metrics())

                    with gr.Column():
                        gr.Markdown("## Available Models")
                        models_info = gr.Markdown(get_model_info())


            # ----------------------------------------------------------------
            # TAB: INFO
            # ----------------------------------------------------------------
            with gr.TabItem("About", id="about"):
                gr.Markdown(f"""
## Pipeline

| Step | What | Script |
|------|------|--------|
| 1 | COCO subset (18 classes) + Open Images V7 (6 classes), remapped IDs | `scripts/build_dataset.py` |
| 2 | Grounding DINO auto-annotation for classes without labels, then review | `scripts/auto_annotate_grounding_dino.py` |
| 3 | Fine-tune YOLO26 n and s with the same recipe | `scripts/train.py` |
| 4 | Global and per-class evaluation | `scripts/evaluate.py` |
| 5 | TensorRT FP16 / INT8 export | `scripts/export_tensorrt.py` |
| 6 | Benchmark on an RTX 5060 Ti and a Jetson Orin Nano | `scripts/benchmark.py` |

### Classes

person, bicycle, car, motorcycle, bus, truck, traffic light, fire hydrant, stop sign,
bench, chair, dog, cat, backpack, umbrella, handbag, suitcase, potted plant,
Door, Stairs, Street light, Traffic sign, Tree, Wheelchair

### Hardware

{hw_summary}

This is a research prototype, not a certified assistive device.
""")

        gr.Markdown("---")
        gr.Markdown(f"*{APP_TITLE} — Ultralytics YOLO26 + Gradio*")

    return app


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    args = parser.parse_args()

    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
