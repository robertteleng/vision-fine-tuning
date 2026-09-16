#!/usr/bin/env python3
"""
Inference script — navigation obstacle detector

Run inference with trained YOLO models on images, videos, or webcam.

Usage:
    python scripts/inference.py --source image.jpg
    python scripts/inference.py --source folder/
    python scripts/inference.py --source video.mp4
    python scripts/inference.py --source 0               # webcam
    python scripts/inference.py --source image.jpg --model models/best.pt
"""

import sys
import shutil
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import find_best_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_inference(args):
    from ultralytics import YOLO

    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            alt_path = PROJECT_ROOT / "models" / args.model
            if alt_path.exists():
                model_path = alt_path
            else:
                logger.error(f"Model not found: {args.model}")
                sys.exit(1)
    else:
        model_path = find_best_model()
        if not model_path:
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
    logger.info(f"Classes: {model.names}")

    source = args.source
    if source.isdigit():
        source = int(source)
        logger.info(f"Source: Webcam {source}")
    elif Path(source).exists():
        p = Path(source)
        if p.is_file():
            logger.info(f"Source: {p.name}")
        else:
            logger.info(f"Source: directory ({len(list(p.glob('*')))} files)")
    else:
        logger.error(f"Source not found: {source}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show=args.show,
        project=str(PROJECT_ROOT / "runs" / "inference"),
        name=f"predict_{timestamp}",
        exist_ok=True,
        verbose=args.verbose,
        device=args.device,
        stream=True,
    )

    total_detections = 0
    processed = 0

    for result in results:
        processed += 1
        boxes = result.boxes
        n = len(boxes)
        total_detections += n

        if n > 0:
            logger.info(f"\n{result.path} — {n} detections")
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
                logger.info(f"  [{i+1}] {name}: {conf:.2%} @ ({x1}, {y1}, {x2}, {y2})")

    logger.info(f"\nProcessed: {processed} | Detections: {total_detections} | "
                f"Avg: {total_detections / max(processed, 1):.1f}/file")

    # Convert .avi to .mp4 (Ultralytics saves uncompressed MJPG on Linux)
    if args.save:
        output_dir = PROJECT_ROOT / "runs" / "inference" / f"predict_{timestamp}"
        for avi_file in output_dir.glob("*.avi"):
            mp4_file = avi_file.with_suffix(".mp4")
            encoder = _detect_ffmpeg_encoder()
            if encoder:
                # NVENC uses -qp for quality, libx264 uses -crf
                quality = ["-qp", "23"] if "nvenc" in encoder else ["-crf", "23"]
                cmd = ["ffmpeg", "-y", "-i", str(avi_file), "-c:v", encoder, *quality, str(mp4_file)]
                logger.info(f"Converting to MP4 ({encoder}): {mp4_file.name}")
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 0:
                    avi_file.unlink()
                    logger.info(f"Saved: {mp4_file} ({mp4_file.stat().st_size / (1024*1024):.1f} MB)")
                else:
                    logger.warning(f"ffmpeg failed, keeping .avi: {avi_file.name}")


def _detect_ffmpeg_encoder() -> str | None:
    """Return best available ffmpeg h264 encoder, or None if ffmpeg missing."""
    if not shutil.which("ffmpeg"):
        return None
    # Try NVENC first (hardware), fall back to libx264 (software)
    for encoder in ("h264_nvenc", "libx264"):
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True,
        )
        if encoder in result.stdout:
            return encoder
    return "libx264"


def export_model(args):
    from ultralytics import YOLO

    model_path = Path(args.model) if args.model else find_best_model()
    if not model_path or not Path(model_path).exists():
        logger.error("No model found for export.")
        sys.exit(1)

    logger.info(f"Exporting: {model_path}")
    model = YOLO(str(model_path))
    export_path = model.export(
        format=args.export_format,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=True,
    )
    logger.info(f"Exported to: {export_path}")


def main():
    parser = argparse.ArgumentParser(description="Navigation detector — Inference")

    parser.add_argument("--source", "-s", default=None, help="Image, directory, video, or webcam index")
    parser.add_argument("--model", "-m", default=None, help="Model path (.pt/.engine/.onnx)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=100, help="Max detections per image")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated results")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO format labels")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence in labels")
    parser.add_argument("--save-crop", action="store_true", help="Save detection crops")
    parser.add_argument("--show", action="store_true", help="Show results in window")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--device", default="0", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--export", action="store_true", help="Export model instead of inference")
    parser.add_argument("--export-format", default="onnx",
                        choices=["onnx", "torchscript", "openvino", "engine", "coreml", "tflite"])
    parser.add_argument("--half", action="store_true", help="FP16 export")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shapes")

    args = parser.parse_args()
    if args.no_save:
        args.save = False

    if args.export:
        export_model(args)
    elif args.source:
        run_inference(args)
    else:
        parser.print_help()
        print("\nSpecify --source or --export")
        sys.exit(1)


if __name__ == "__main__":
    main()
