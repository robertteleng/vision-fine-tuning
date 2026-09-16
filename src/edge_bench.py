"""Benchmark protocol shared by every machine (x86 + RTX and Jetson).

The rules live in docs/BENCHMARK_METHODOLOGY.md. This module holds the parts
that must not drift between machines: latency statistics, the INT8
calibration split, artifact names, the result record and the INT8 decision
criterion. It has no GPU dependency so it can be unit-tested anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

SCHEMA_VERSION = 1
PRECISIONS = ("fp32", "fp16", "int8")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# Fixed evaluation settings. iou=0.7 is what training validated with, so the
# numbers are comparable with the training curves.
EVAL_SETTINGS = {"imgsz": 640, "batch": 1, "conf": 0.001, "iou": 0.7}

# Written on 2026-09-16, before any INT8 number was measured. INT8 replaces
# FP16 on a device only if all three hold on that device.
INT8_CRITERION = {
    "decided_on": "2026-09-16",
    "min_p95_latency_reduction": 0.20,
    "max_map50_drop": 0.01,
    "max_stairs_map50_drop": 0.02,
}


# --------------------------------------------------------------------------- latency


def latency_summary(samples_ms: Sequence[float]) -> dict:
    """Mean, spread and tail percentiles of per-image latencies in milliseconds."""
    if len(samples_ms) == 0:
        raise ValueError("no latency samples")
    a = np.asarray(samples_ms, dtype=float)
    mean = float(a.mean())
    return {
        "n": int(a.size),
        "mean": mean,
        "std": float(a.std()),
        "min": float(a.min()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "fps": 1000.0 / mean if mean > 0 else float("inf"),
    }


# --------------------------------------------------------------------------- data


def list_images(directory: Path) -> list[Path]:
    """Images in a directory, sorted by name so every machine sees the same order."""
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def sample_images(images: Sequence[Path], n: int, seed: int) -> list[Path]:
    """Deterministic sample (sorted input + seeded RNG). Returns all if n >= len."""
    images = sorted(images)
    if n >= len(images):
        return list(images)
    return sorted(random.Random(seed).sample(images, n))


def dataset_split_dir(dataset_yaml: Path, split: str) -> Path:
    cfg = yaml.safe_load(Path(dataset_yaml).read_text())
    return Path(cfg["path"]) / cfg[split]


def write_calibration_yaml(dataset_yaml: Path, out_dir: Path, n: int = 1000, seed: int = 0) -> Path:
    """Dataset YAML whose ``val`` split is a sample of TRAIN images.

    Ultralytics calibrates INT8 on the ``val`` split of the YAML it is given.
    Calibrating on the images the model is later scored on would leak the
    evaluation set into the quantization ranges, so calibration uses train.
    """
    cfg = yaml.safe_load(Path(dataset_yaml).read_text())
    train_images = list_images(Path(cfg["path"]) / cfg["train"])
    chosen = sample_images(train_images, n, seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = out_dir / f"calibration_train_{len(chosen)}_seed{seed}.txt"
    listing.write_text("\n".join(str(p.resolve()) for p in chosen) + "\n")
    calib = {"path": str(out_dir.resolve()), "train": listing.name, "val": listing.name, "names": cfg["names"]}
    calib_yaml = out_dir / f"calibration_train_{len(chosen)}_seed{seed}.yaml"
    calib_yaml.write_text(yaml.safe_dump(calib, sort_keys=False))
    return calib_yaml


# --------------------------------------------------------------------------- artifacts


def host_tag() -> str:
    """Short machine tag used in artifact names: 'jetson' or the hostname."""
    return "jetson" if is_jetson() else re.sub(r"[^a-z0-9]+", "-", socket.gethostname().lower()).strip("-")


def engine_path(weights: Path, precision: str, host: str, engines_dir: Path) -> Path:
    """Engines are device-specific, so the host is part of the name."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}")
    return Path(engines_dir) / f"{Path(weights).stem}_{precision}_{host}.engine"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- accuracy


def accuracy_summary(metrics, names: dict[int, str]) -> dict:
    """Global and per-class box metrics from an Ultralytics ``DetMetrics``.

    Classes absent from the evaluated split have no AP and are left out
    rather than reported as zero.
    """
    box = metrics.box
    per_class = {
        names[int(c)]: {"map50": float(box.ap50[i]), "map50_95": float(box.ap[i])}
        for i, c in enumerate(box.ap_class_index)
    }
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- environment


def is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").exists()


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _version(module: str) -> str | None:
    try:
        return __import__(module).__version__
    except Exception:
        return None


def environment() -> dict:
    """Everything needed to tell whether two numbers are comparable."""
    env = {
        "hostname": socket.gethostname(),
        "host_tag": host_tag(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": _version("torch"),
        "ultralytics": _version("ultralytics"),
        "tensorrt": _version("tensorrt"),
        "onnx": _version("onnx"),
        "numpy": np.__version__,
    }
    try:
        import torch

        env["cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        env.update(cuda=None, cudnn=None, gpu=None)
    env["nvidia_driver"] = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if is_jetson():
        env["l4t_release"] = Path("/etc/nv_tegra_release").read_text().splitlines()[0].strip()
        env["power_mode"] = _run(["nvpmodel", "-q"])
    env["cpu"] = _cpu_model()
    env["git_commit"] = _run(["git", "-C", str(Path(__file__).parent.parent), "rev-parse", "HEAD"])
    env["git_dirty"] = bool(_run(["git", "-C", str(Path(__file__).parent.parent), "status", "--porcelain", "--untracked-files=no"]))
    return env


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def memory_used_mb() -> float | None:
    """GPU memory in use (discrete GPU) or system RAM in use (Jetson, unified memory)."""
    if is_jetson():
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0])
        return (info["MemTotal"] - info["MemAvailable"]) / 1024
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    return float(out.splitlines()[0]) if out else None


# --------------------------------------------------------------------------- records


def make_record(*, environment: dict, command: Sequence[str], model: dict, protocol: dict,
                latency_ms: dict | None, accuracy: dict | None, memory: dict | None) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment,
        "command": list(command),
        "model": model,
        "protocol": protocol,
        "latency_ms": latency_ms,
        "accuracy": accuracy,
        "memory": memory,
    }


def record_filename(record: dict) -> str:
    stamp = record["created"].replace(":", "").replace("-", "")[:15]
    m = record["model"]
    return f"{stamp}_{record['environment']['host_tag']}_{m['name']}_{m['precision']}.json"


def write_record(record: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / record_filename(record)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def load_records(results_dir: Path) -> list[dict]:
    records = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))]
    return [r for r in records if r.get("schema") == SCHEMA_VERSION]


def latest_by_key(records: Sequence[dict]) -> dict[tuple[str, str, str], dict]:
    """Keep the newest record per (host, model, precision)."""
    latest: dict[tuple[str, str, str], dict] = {}
    for r in sorted(records, key=lambda r: r["created"]):
        latest[(r["environment"]["host_tag"], r["model"]["name"], r["model"]["precision"])] = r
    return latest


# --------------------------------------------------------------------------- INT8 decision


def int8_verdict(fp16: dict, int8: dict, criterion: dict = INT8_CRITERION, key_class: str = "Stairs") -> dict:
    """Apply the pre-registered criterion to one device's FP16 and INT8 records."""
    p95_fp16 = fp16["latency_ms"]["end_to_end"]["p95"]
    p95_int8 = int8["latency_ms"]["end_to_end"]["p95"]
    reduction = (p95_fp16 - p95_int8) / p95_fp16
    map_drop = fp16["accuracy"]["map50"] - int8["accuracy"]["map50"]
    cls_drop = (fp16["accuracy"]["per_class"][key_class]["map50"]
                - int8["accuracy"]["per_class"][key_class]["map50"])
    checks = {
        "p95_latency_reduction": (reduction, reduction >= criterion["min_p95_latency_reduction"]),
        "map50_drop": (map_drop, map_drop <= criterion["max_map50_drop"]),
        f"{key_class.lower()}_map50_drop": (cls_drop, cls_drop <= criterion["max_stairs_map50_drop"]),
    }
    return {
        "checks": {k: {"value": v, "pass": ok} for k, (v, ok) in checks.items()},
        "use_int8": all(ok for _, ok in checks.values()),
        "criterion": criterion,
    }


# --------------------------------------------------------------------------- tables


def markdown_table(records: Sequence[dict], classes: Sequence[str] = ("Stairs", "Door", "person")) -> str:
    """One row per (host, model, precision), newest record wins. Generated, never hand-edited."""
    header = ["Device", "Model", "Precision", "Mean ms", "p95 ms", "FPS", "Size MB", "mAP50", "mAP50-95", *classes]
    rows = []
    for (host, name, precision), r in sorted(latest_by_key(records).items(),
                                             key=lambda kv: (kv[0][0], kv[0][1], PRECISIONS.index(kv[0][2]))):
        lat = (r.get("latency_ms") or {}).get("end_to_end")
        acc = r.get("accuracy")
        rows.append([
            r["environment"].get("gpu") or host,
            name,
            precision.upper(),
            f"{lat['mean']:.2f}" if lat else "—",
            f"{lat['p95']:.2f}" if lat else "—",
            f"{lat['fps']:.0f}" if lat else "—",
            f"{r['model']['artifact_mb']:.1f}" if r["model"].get("artifact_mb") is not None else "—",
            f"{acc['map50']:.3f}" if acc else "—",
            f"{acc['map50_95']:.3f}" if acc else "—",
            *[f"{acc['per_class'][c]['map50']:.3f}" if acc and c in acc["per_class"] else "—" for c in classes],
        ])
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"
