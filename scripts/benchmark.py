#!/usr/bin/env python3
"""
Benchmark script — Fine-Tuning Studio

Compare inference speed across model formats (PyTorch, ONNX, TensorRT).

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --model models/best.pt
    python scripts/benchmark.py --iterations 200
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import find_best_model
from src.hardware import detect_gpu, get_hardware_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def benchmark_model(model_path: Path, imgsz: int, iterations: int, warmup: int):
    """Run benchmark on a single model."""
    from ultralytics import YOLO
    import torch

    logger.info(f"Loading: {model_path.name}")
    model = YOLO(str(model_path))

    dummy_img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    logger.info(f"  Warmup ({warmup} iterations)...")
    for _ in range(warmup):
        model(dummy_img, verbose=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"  Benchmark ({iterations} iterations)...")
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        model(dummy_img, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times_ms = np.array(times) * 1000
    return {
        "mean": np.mean(times_ms),
        "std": np.std(times_ms),
        "min": np.min(times_ms),
        "max": np.max(times_ms),
        "median": np.median(times_ms),
        "fps": 1000 / np.mean(times_ms),
    }


def run_benchmark(args):
    """Run comparative benchmark."""
    gpu = detect_gpu()

    logger.info("=" * 60)
    logger.info("INFERENCE BENCHMARK")
    logger.info("=" * 60)
    logger.info(get_hardware_summary(gpu))
    logger.info(f"Image size: {args.imgsz}x{args.imgsz}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info("-" * 60)

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

    models_to_test = []

    if model_path.suffix == ".pt":
        models_to_test.append(("PyTorch", model_path))

    parent = model_path.parent
    stem = model_path.stem

    onnx_path = parent / f"{stem}.onnx"
    if onnx_path.exists():
        models_to_test.append(("ONNX", onnx_path))

    engine_path = parent / f"{stem}.engine"
    if engine_path.exists():
        models_to_test.append(("TensorRT", engine_path))

    if not models_to_test:
        models_to_test.append(("Model", model_path))

    results = {}
    for name, path in models_to_test:
        try:
            logger.info(f"\n{name}:")
            stats = benchmark_model(path, args.imgsz, args.iterations, args.warmup)
            results[name] = stats
            logger.info(f"  Mean: {stats['mean']:.2f}ms (+/-{stats['std']:.2f}ms)")
            logger.info(f"  FPS: {stats['fps']:.1f}")
        except Exception as e:
            logger.warning(f"  Error: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Format':<12} {'Mean (ms)':<12} {'Std (ms)':<10} {'FPS':<8} {'Min (ms)':<10} {'Max (ms)'}")
    logger.info("-" * 60)

    baseline_fps = None
    for name, stats in results.items():
        if baseline_fps is None:
            baseline_fps = stats["fps"]
            speedup = ""
        else:
            speedup = f" ({stats['fps'] / baseline_fps:.1f}x)"

        logger.info(
            f"{name:<12} {stats['mean']:>8.2f}    {stats['std']:>8.2f}  "
            f"{stats['fps']:>6.1f}{speedup:<6} {stats['min']:>8.2f}    {stats['max']:.2f}"
        )

    logger.info("=" * 60)

    if len(results) == 1 and "PyTorch" in results:
        logger.info("\nTo compare formats, export first:")
        logger.info("  python scripts/export_tensorrt.py --format engine --half")
        logger.info("  python scripts/export_tensorrt.py --format onnx")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning Studio — Benchmark")
    parser.add_argument("--model", "-m", default=None, help="Model .pt path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--iterations", "-n", type=int, default=100, help="Iterations")
    parser.add_argument("--warmup", "-w", type=int, default=10, help="Warmup iterations")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
