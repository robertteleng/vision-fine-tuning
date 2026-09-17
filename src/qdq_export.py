"""Explicit INT8: a Q/DQ ONNX from NVIDIA ModelOpt, then a TensorRT engine per device.

Why not Ultralytics' ``int8=True``: that path uses TensorRT implicit
quantization (deprecated since TensorRT 10). On the RTX 5060 Ti it built INT8
engines slower than FP16, because layers without INT8 kernels fell back to
FP32. On Jetson (TensorRT 10.3) calibration crashes with an internal
``checkSanity`` assertion.

Here the INT8 scales live inside the ONNX as QuantizeLinear/DequantizeLinear
nodes. They are computed once, on x86, from the same TRAIN calibration images,
and every device builds its engine from that same file with INT8 + FP16
enabled. Layers left unquantized run in FP16, not FP32, and no device
calibrates at build time.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from src.edge_bench import EVAL_SETTINGS, engine_path, host_tag, list_images, sample_images

CALIBRATION_METHOD = "entropy"  # ModelOpt default for INT8


def qdq_onnx_path(weights: Path, qdq_dir: Path) -> Path:
    return Path(qdq_dir) / f"{Path(weights).stem}_int8_qdq.onnx"


def letterbox_tensor(image_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """Same pre-processing as Ultralytics predict on a static engine: letterbox, RGB, CHW, [0, 1]."""
    from ultralytics.data.augment import LetterBox

    padded = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image_bgr)
    return np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0


def calibration_array(dataset_yaml: Path, n: int, seed: int, imgsz: int) -> tuple[np.ndarray, list[str]]:
    """N x 3 x imgsz x imgsz float32 batch from TRAIN images (same sample rule as the implicit path)."""
    import cv2

    cfg = yaml.safe_load(Path(dataset_yaml).read_text())
    paths = sample_images(list_images(Path(cfg["path"]) / cfg["train"]), n, seed)
    batch = np.stack([letterbox_tensor(cv2.imread(str(p)), imgsz) for p in paths])
    return batch, [p.name for p in paths]


def quantize(weights: Path, dataset_yaml: Path, qdq_dir: Path, calib_images: int = 1000, seed: int = 0) -> dict:
    """Export ONNX with Ultralytics and insert Q/DQ nodes with ModelOpt (x86 + CUDA)."""
    from modelopt.onnx.quantization import quantize as modelopt_quantize
    import onnx
    from ultralytics import YOLO

    out = qdq_onnx_path(weights, qdq_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="qdq_") as scratch:
        local = Path(scratch) / Path(weights).name
        shutil.copy2(weights, local)
        fp32_onnx = Path(YOLO(str(local)).export(format="onnx", imgsz=EVAL_SETTINGS["imgsz"], batch=1,
                                                   dynamic=False, simplify=True, half=False, verbose=False))
        calib, names = calibration_array(dataset_yaml, calib_images, seed, EVAL_SETTINGS["imgsz"])
        modelopt_quantize(str(fp32_onnx), quantize_mode="int8", calibration_data=calib,
                          calibration_method=CALIBRATION_METHOD, calibration_eps=["cuda:0", "cpu"],
                          high_precision_dtype="fp16", output_path=str(out))
        # Keep Ultralytics' metadata (names, stride, imgsz...) so the engine loads like any other.
        source_meta = {p.key: p.value for p in onnx.load(str(fp32_onnx), load_external_data=False).metadata_props}
    model = onnx.load(str(out))
    del model.metadata_props[:]
    for key, value in source_meta.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, value
    onnx.save(model, str(out))
    info = {"qdq_onnx": str(out), "calibration": {"source": "train", "images": len(names), "seed": seed,
            "method": f"ModelOpt {CALIBRATION_METHOD}", "high_precision_dtype": "fp16"},
            "quantize_seconds": time.perf_counter() - start}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def build_engine(qdq_onnx: Path, weights: Path, engines_dir: Path, workspace_gb: float = 4.0,
                 rebuild: bool = False) -> dict:
    """Build the device engine from the shared Q/DQ ONNX with INT8 + FP16 enabled."""
    import onnx
    import tensorrt as trt

    target = engine_path(weights, "int8_qdq", host_tag(), engines_dir)
    info = {"artifact": str(target), "source_onnx": str(qdq_onnx), "builder_flags": ["INT8", "FP16"],
            "fallback_precision": "FP16", "export_seconds": None, "reused": target.exists() and not rebuild}
    sidecar = Path(qdq_onnx).with_suffix(".json")
    if sidecar.exists():
        info["calibration"] = json.loads(sidecar.read_text())["calibration"]
    if info["reused"]:
        return info

    class ErrorLogger(trt.ILogger):
        """Keeps TensorRT's error messages so a failed build records why it failed."""

        def __init__(self):
            trt.ILogger.__init__(self)
            self.errors: list[str] = []

        def log(self, severity, msg):
            if severity in (trt.ILogger.INTERNAL_ERROR, trt.ILogger.ERROR):
                self.errors.append(msg)

    logger = ErrorLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(qdq_onnx).read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT {trt.__version__} engine build failed: " + " | ".join(logger.errors[-3:]))
    info["export_seconds"] = time.perf_counter() - start

    meta = {p.key: p.value for p in onnx.load(str(qdq_onnx), load_external_data=False).metadata_props}
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:  # Ultralytics engine layout: 4-byte length, JSON metadata, engine
        blob = json.dumps(meta).encode()
        f.write(len(blob).to_bytes(4, byteorder="little", signed=True))
        f.write(blob)
        f.write(serialized)
    return info
