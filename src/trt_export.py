"""TensorRT export with the pitfalls of Ultralytics 8.4 handled explicitly.

- Engines are built in a scratch directory and moved to a device-specific
  name, so exporting never overwrites files next to the weights.
- INT8 calibrates on a sample of TRAIN images (see
  ``edge_bench.write_calibration_yaml``). Ultralytics would otherwise use the
  validation split, the same images the engine is scored on.
- Ultralytics reuses ``<name>.cache`` if it exists. A stale cache from a
  different calibration set would silently change the INT8 engine, so the
  scratch directory always starts empty.
- Ultralytics builds INT8 engines with only the INT8 builder flag: layers
  TensorRT cannot quantize run in FP32, not FP16. That is part of what is
  being measured, so it is recorded rather than patched.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from src.edge_bench import EVAL_SETTINGS, PRECISIONS, engine_path, host_tag, write_calibration_yaml


def export_engine(weights: Path, precision: str, dataset_yaml: Path, engines_dir: Path,
                  calib_images: int = 1000, seed: int = 0, workspace_gb: float = 4.0,
                  rebuild: bool = False) -> dict:
    """Build (or reuse) the TensorRT engine for one precision. Returns artifact metadata."""
    if precision not in PRECISIONS or precision == "fp32":
        raise ValueError(f"engine precision must be fp16 or int8, got {precision!r}")
    from ultralytics import YOLO

    weights = Path(weights)
    target = engine_path(weights, precision, host_tag(), engines_dir)
    info = {"artifact": str(target), "calibration": None, "export_seconds": None, "reused": target.exists() and not rebuild}
    if precision == "int8":
        info["calibration"] = {"source": "train", "images": calib_images, "seed": seed,
                               "algorithm": "TensorRT MinMax (Ultralytics default)",
                               "builder_flags": ["INT8"], "fallback_precision": "FP32"}
    if info["reused"]:
        return info

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trt_export_") as scratch:
        scratch = Path(scratch)
        local_weights = scratch / weights.name
        shutil.copy2(weights, local_weights)
        kwargs = dict(format="engine", imgsz=EVAL_SETTINGS["imgsz"], batch=1, device=0,
                      dynamic=False, simplify=True, workspace=workspace_gb, verbose=False)
        if precision == "fp16":
            kwargs["half"] = True
        else:
            kwargs["int8"] = True
            kwargs["data"] = str(write_calibration_yaml(dataset_yaml, scratch / "calibration", calib_images, seed))
        start = time.perf_counter()
        built = Path(YOLO(str(local_weights)).export(**kwargs))
        info["export_seconds"] = time.perf_counter() - start
        shutil.move(str(built), target)
    return info
