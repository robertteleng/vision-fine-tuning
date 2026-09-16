#!/usr/bin/env python3
"""
Benchmark — step 6 of the pipeline.

For every weights x precision: build the artifact (PyTorch for FP32, TensorRT
engine for FP16/INT8, built on this machine), score it on the full validation
split, time it on real validation images, and write one JSON record with the
environment. The protocol is in docs/BENCHMARK_METHODOLOGY.md.

Usage:
    uv sync --extra export
    uv run python scripts/benchmark.py \
        --weights models/yolo26n_nav.pt models/yolo26s_nav.pt \
        --precisions fp32 fp16 int8
    uv run python scripts/make_tables.py
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import edge_bench as eb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

LATENCY_CONF = 0.25  # a deployment-like threshold: post-processing cost depends on it


def gpu_is_busy() -> list[str]:
    """Other compute processes on a discrete GPU (not available on Jetson)."""
    if eb.is_jetson():
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except OSError:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def load_images(paths):
    import cv2

    images = [cv2.imread(str(p)) for p in paths]
    missing = [p for p, im in zip(paths, images) if im is None]
    if missing:
        raise FileNotFoundError(f"could not read {len(missing)} images, e.g. {missing[0]}")
    return images


def measure_latency(model, images, warmup: int, half: bool) -> dict:
    import torch

    def run(img):
        return model.predict(img, imgsz=eb.EVAL_SETTINGS["imgsz"], conf=LATENCY_CONF, device=0,
                             half=half, verbose=False)

    for i in range(warmup):
        run(images[i % len(images)])
    torch.cuda.synchronize()

    e2e, pre, inf, post = [], [], [], []
    for img in images:
        start = time.perf_counter()
        result = run(img)
        torch.cuda.synchronize()
        e2e.append((time.perf_counter() - start) * 1000)
        pre.append(result[0].speed["preprocess"])
        inf.append(result[0].speed["inference"])
        post.append(result[0].speed["postprocess"])
    return {
        "end_to_end": eb.latency_summary(e2e),
        "preprocess": eb.latency_summary(pre),
        "inference": eb.latency_summary(inf),
        "postprocess": eb.latency_summary(post),
    }


def evaluate(model, dataset_yaml: Path, half: bool, scratch: Path) -> dict:
    metrics = model.val(data=str(dataset_yaml), split="val", device=0, half=half, plots=False, verbose=False,
                        project=str(scratch), name="val", exist_ok=True, **eb.EVAL_SETTINGS)
    return eb.accuracy_summary(metrics, model.names)


def main():
    parser = argparse.ArgumentParser(description="Navigation detector — benchmark")
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--precisions", nargs="+", choices=eb.PRECISIONS, default=list(eb.PRECISIONS))
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/nav_combined/dataset.yaml")
    parser.add_argument("--latency-images", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--calib-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engines-dir", type=Path, default=PROJECT_ROOT / "models/engines")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "benchmarks/results")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild engines even if they exist")
    parser.add_argument("--allow-busy-gpu", action="store_true", help="Measure even if other processes use the GPU")
    args = parser.parse_args()

    busy = gpu_is_busy()
    if busy and not args.allow_busy_gpu:
        logger.error(f"GPU in use by {busy}: latency would be contaminated. Stop them or pass --allow-busy-gpu.")
        sys.exit(1)

    import torch
    from ultralytics import YOLO

    from src.trt_export import export_engine

    if not torch.cuda.is_available():
        logger.error("CUDA is required.")
        sys.exit(1)

    environment = eb.environment()
    val_dir = eb.dataset_split_dir(args.data, "val")
    latency_paths = eb.sample_images(eb.list_images(val_dir), args.latency_images, args.seed)
    images = load_images(latency_paths)
    logger.info(f"{environment['gpu']} | {len(images)} latency images preloaded | commit {environment['git_commit']}")

    scratch = PROJECT_ROOT / "runs" / "benchmark"
    for weights in args.weights:
        for precision in args.precisions:
            logger.info(f"=== {weights.stem} {precision.upper()} ===")
            if precision == "fp32":
                artifact = {"artifact": str(weights), "calibration": None, "export_seconds": None, "reused": True}
            else:
                artifact = export_engine(weights, precision, args.data, args.engines_dir,
                                         args.calib_images, args.seed, rebuild=args.rebuild)
            path = Path(artifact["artifact"])
            half = False  # FP32 runs as FP32; engine precision is fixed at build time

            memory_before = eb.memory_used_mb()
            model = YOLO(str(path), task="detect")
            accuracy = None if args.skip_accuracy else evaluate(model, args.data, half, scratch)
            latency = measure_latency(model, images, args.warmup, half)
            memory_after = eb.memory_used_mb()

            record = eb.make_record(
                environment=environment,
                command=sys.argv,
                model={"name": weights.stem, "precision": precision, "weights": str(weights),
                       "weights_sha256": eb.sha256(weights), "artifact_mb": path.stat().st_size / 2**20, **artifact},
                protocol={"eval": eb.EVAL_SETTINGS, "latency_images": len(images), "latency_seed": args.seed,
                          "latency_conf": LATENCY_CONF, "warmup": args.warmup, "dataset": str(args.data)},
                latency_ms=latency,
                accuracy=accuracy,
                memory={"used_before_mb": memory_before, "used_after_mb": memory_after,
                        "kind": "system RAM (unified)" if eb.is_jetson() else "GPU memory (nvidia-smi)"},
            )
            out = eb.write_record(record, args.out)
            e2e = latency["end_to_end"]
            acc = f" | mAP50 {accuracy['map50']:.3f}" if accuracy else ""
            logger.info(f"mean {e2e['mean']:.2f} ms | p95 {e2e['p95']:.2f} ms | {e2e['fps']:.0f} FPS{acc} -> {out.name}")
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
