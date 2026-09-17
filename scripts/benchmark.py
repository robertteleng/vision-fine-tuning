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


def engine_only_latency(engine_file: Path, iterations: int, warmup: int) -> dict:
    """The TensorRT engine alone: input already on the GPU, no pre/post-processing, no Python pipeline.

    Separates what the network costs from what the surrounding code costs.
    """
    import tensorrt as trt
    import torch

    blob = Path(engine_file).read_bytes()
    meta_len = int.from_bytes(blob[:4], "little")
    plan = blob[4 + meta_len:] if 0 < meta_len < 1 << 16 and blob[4:5] == b"{" else blob
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    torch_dtype = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
                   trt.int64: torch.int64, trt.bool: torch.bool, trt.int8: torch.int8}
    tensors = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        tensors[name] = torch.zeros(tuple(engine.get_tensor_shape(name)),
                                    dtype=torch_dtype[engine.get_tensor_dtype(name)], device="cuda")
        context.set_tensor_address(name, tensors[name].data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    for name, t in tensors.items():
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            t.uniform_(0, 1) if t.is_floating_point() else None
    for _ in range(warmup):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        context.execute_async_v3(stream)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return eb.latency_summary(samples)


def evaluate(model, dataset_yaml: Path, half: bool, scratch: Path) -> dict:
    metrics = model.val(data=str(dataset_yaml), split="val", device=0, half=half, plots=False, verbose=False,
                        project=str(scratch), name="val", exist_ok=True, **eb.EVAL_SETTINGS)
    return eb.accuracy_summary(metrics, model.names)


def build_artifact(weights: Path, precision: str, args) -> dict:
    if precision == "fp32":
        return {"artifact": str(weights), "calibration": None, "export_seconds": None, "reused": True}
    if precision == "int8_qdq":
        from src import qdq_export

        qdq = qdq_export.qdq_onnx_path(weights, args.qdq_dir)
        if not qdq.exists() or (args.rebuild and not eb.is_jetson()):
            try:
                qdq_export.quantize(weights, args.data, args.qdq_dir, args.calib_images, args.seed)
            except ImportError as exc:
                raise RuntimeError(f"{qdq} missing; create it on x86 with `uv sync --extra quantize`") from exc
        return qdq_export.build_engine(qdq, weights, args.engines_dir, rebuild=args.rebuild)
    from src.trt_export import export_engine

    return export_engine(weights, precision, args.data, args.engines_dir, args.calib_images, args.seed,
                         rebuild=args.rebuild)


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
    parser.add_argument("--qdq-dir", type=Path, default=PROJECT_ROOT / "models/qdq",
                        help="Shared Q/DQ ONNX files (created on x86 with --extra quantize)")
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
            try:
                artifact = build_artifact(weights, precision, args)
            except Exception as exc:  # a failed build is recorded, the run goes on
                logger.error(f"{weights.stem} {precision}: build failed: {exc}")
                record = eb.make_record(
                    environment=environment, command=sys.argv,
                    model={"name": weights.stem, "precision": precision, "weights": eb.repo_relative(weights),
                           "weights_sha256": eb.sha256(weights)},
                    protocol={"eval": eb.EVAL_SETTINGS, "dataset": eb.repo_relative(args.data)},
                    latency_ms=None, accuracy=None, memory=None, status="build_failed", error=str(exc)[:2000])
                logger.info(f"-> {eb.write_record(record, args.out).name}")
                continue
            path = Path(artifact["artifact"])
            half = False  # FP32 runs as FP32; engine precision is fixed at build time

            memory_before = eb.memory_used_mb()
            model = YOLO(str(path), task="detect")
            accuracy = None if args.skip_accuracy else evaluate(model, args.data, half, scratch)
            latency = measure_latency(model, images, args.warmup, half)
            if path.suffix == ".engine":
                latency["engine_only"] = engine_only_latency(path, len(images), args.warmup)
            memory_after = eb.memory_used_mb()

            record = eb.make_record(
                environment=environment,
                command=sys.argv,
                model={"name": weights.stem, "precision": precision, "weights": eb.repo_relative(weights),
                       "weights_sha256": eb.sha256(weights), "artifact_mb": path.stat().st_size / 2**20,
                       **{k: eb.repo_relative(v) if k in ("artifact", "source_onnx") else v for k, v in artifact.items()}},
                protocol={"eval": eb.EVAL_SETTINGS, "latency_images": len(images), "latency_seed": args.seed,
                          "latency_conf": LATENCY_CONF, "warmup": args.warmup, "dataset": eb.repo_relative(args.data)},
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
