"""Build the 24-class navigation dataset from COCO 2017 and Open Images V7.

Why a combined dataset: fine-tuning replaces the YOLO detection head, so a
model trained only on the new classes forgets the COCO ones. Every class the
detector needs has to be in one dataset, with non-overlapping IDs.

The images are fixed by the manifests in ``data_manifest/`` (one image ID per
line). Labels are converted here from the original annotation files, not from
an intermediate export, so the result depends only on the manifests and the
upstream annotations.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light",
    "fire hydrant", "stop sign", "bench", "chair", "dog", "cat", "backpack",
    "umbrella", "handbag", "suitcase", "potted plant",
)
# Classes COCO does not have. They take IDs 18-23, after the COCO ones.
OPENIMAGES_CLASSES = ("Door", "Stairs", "Street light", "Traffic sign", "Tree", "Wheelchair")
CLASS_NAMES = COCO_CLASSES + OPENIMAGES_CLASSES

SPLITS = ("train", "val")
# FiftyOne names the validation split "validation" on disk.
FIFTYONE_SPLIT = {"train": "train", "val": "validation"}

COCO_PREFIX = "coco_"
OPENIMAGES_PREFIX = "custom_"


@dataclass(frozen=True)
class Sample:
    """One image of the combined dataset and its YOLO label lines."""

    split: str
    name: str  # file stem inside the combined dataset, e.g. "coco_000000000030"
    image_path: Path
    lines: tuple[str, ...]


def read_manifest(path: Path) -> list[str]:
    """Image IDs, one per line; blank lines ignored. Fails on duplicates."""
    ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    duplicated = [i for i, n in Counter(ids).items() if n > 1]
    if duplicated:
        raise ValueError(f"{path}: duplicated image IDs {duplicated[:5]}")
    return ids


def format_yolo_line(class_id: int, cx: float, cy: float, w: float, h: float) -> str:
    """YOLO label line with 6 decimals (the precision FiftyOne exports)."""
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def coco_labels(instances: dict, image_ids: Iterable[int]) -> dict[int, list[str]]:
    """YOLO lines per COCO image, keeping only ``COCO_CLASSES``.

    Crowd annotations (``iscrowd=1``) are kept: they were part of the training
    data. Raises if a requested image is missing or has no kept annotation,
    since the manifest only lists images with at least one target class.
    """
    wanted = {int(i) for i in image_ids}
    category = {c["id"]: c["name"] for c in instances["categories"]}
    images = {im["id"]: im for im in instances["images"] if im["id"] in wanted}
    missing = wanted - images.keys()
    if missing:
        raise KeyError(f"{len(missing)} COCO image IDs not in annotations, e.g. {sorted(missing)[:5]}")

    lines: dict[int, list[str]] = {i: [] for i in wanted}
    for ann in instances["annotations"]:
        image_id = ann["image_id"]
        name = category[ann["category_id"]]
        if image_id not in wanted or name not in COCO_CLASSES:
            continue
        im = images[image_id]
        x, y, w, h = ann["bbox"]
        lines[image_id].append(format_yolo_line(
            COCO_CLASSES.index(name),
            (x + w / 2) / im["width"], (y + h / 2) / im["height"],
            w / im["width"], h / im["height"],
        ))
    _require_labels(lines, "COCO")
    return lines


def openimages_class_names(class_descriptions_csv: Path) -> dict[str, str]:
    """Map Open Images label MIDs (e.g. ``/m/02dgv``) to display names."""
    with open(class_descriptions_csv, newline="") as f:
        return {row[0]: row[1] for row in csv.reader(f) if len(row) >= 2}


def openimages_labels(detections_csv: Path, mid_to_name: dict[str, str],
                      image_ids: Iterable[str]) -> dict[str, list[str]]:
    """YOLO lines per Open Images image, keeping only ``OPENIMAGES_CLASSES``.

    Class IDs are shifted after the COCO ones (Door -> 18 ... Wheelchair -> 23).
    ``IsGroupOf`` boxes are kept. The CSV is streamed: the train file is ~2 GB.
    """
    wanted = set(image_ids)
    offset = len(COCO_CLASSES)
    lines: dict[str, list[str]] = {i: [] for i in wanted}
    with open(detections_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["ImageID"] not in wanted:
                continue
            name = mid_to_name.get(row["LabelName"])
            if name not in OPENIMAGES_CLASSES:
                continue
            x0, x1 = float(row["XMin"]), float(row["XMax"])
            y0, y1 = float(row["YMin"]), float(row["YMax"])
            lines[row["ImageID"]].append(format_yolo_line(
                offset + OPENIMAGES_CLASSES.index(name),
                (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0,
            ))
    _require_labels(lines, "Open Images")
    return lines


def _require_labels(lines: dict, source: str) -> None:
    empty = [i for i, ls in lines.items() if not ls]
    if empty:
        raise ValueError(f"{len(empty)} {source} images without target classes, e.g. {sorted(empty)[:5]}")


def missing_images(samples: Iterable[Sample]) -> list[Path]:
    return [s.image_path for s in samples if not s.image_path.exists()]


def write_dataset(samples: Iterable[Sample], out_dir: Path, link: str = "hard") -> None:
    """Write ``images/{split}`` and ``labels/{split}`` plus ``dataset.yaml``.

    ``link``: ``hard`` (hard link, falls back to copy across filesystems),
    ``symlink`` or ``copy``.
    """
    out_dir = Path(out_dir)
    for sample in samples:
        image_dst = out_dir / "images" / sample.split / f"{sample.name}{sample.image_path.suffix}"
        label_dst = out_dir / "labels" / sample.split / f"{sample.name}.txt"
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        _place(sample.image_path, image_dst, link)
        label_dst.write_text("\n".join(sample.lines) + "\n")
    write_dataset_yaml(out_dir)


def _place(src: Path, dst: Path, link: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link == "symlink":
        dst.symlink_to(Path(src).resolve())
    elif link == "hard":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif link == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown link mode: {link}")


def write_dataset_yaml(out_dir: Path) -> Path:
    """Ultralytics dataset file. ``path`` is absolute so training works from any cwd."""
    out_dir = Path(out_dir)
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    text = (
        "# Navigation dataset: 18 COCO classes + 6 Open Images V7 classes.\n"
        "# Generated by scripts/build_dataset.py from data_manifest/. Do not edit.\n"
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names}\n"
    )
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(text)
    return yaml_path


def dataset_stats(samples: Iterable[Sample]) -> dict:
    """Images and instances per class and split: the numbers for the dataset card."""
    stats = {split: {"images": 0, "instances": Counter(), "images_with": Counter()} for split in SPLITS}
    for sample in samples:
        s = stats[sample.split]
        s["images"] += 1
        ids = [int(line.split()[0]) for line in sample.lines]
        s["instances"].update(ids)
        s["images_with"].update(set(ids))
    return {
        split: {
            "images": s["images"],
            "classes": {
                CLASS_NAMES[i]: {"instances": s["instances"][i], "images": s["images_with"][i]}
                for i in range(len(CLASS_NAMES))
            },
        }
        for split, s in stats.items()
    }


def compare_labels(expected_dir: Path, actual_dir: Path, tol: float = 1e-4) -> list[str]:
    """Differences between two YOLO label trees, allowing float rounding."""
    expected_dir, actual_dir = Path(expected_dir), Path(actual_dir)
    diffs = []
    expected = {p.relative_to(expected_dir) for p in expected_dir.rglob("*.txt")}
    actual = {p.relative_to(actual_dir) for p in actual_dir.rglob("*.txt")}
    diffs += [f"missing: {p}" for p in sorted(expected - actual)]
    diffs += [f"extra: {p}" for p in sorted(actual - expected)]
    for rel in sorted(expected & actual):
        a = sorted(_parse(expected_dir / rel))
        b = sorted(_parse(actual_dir / rel))
        if len(a) != len(b) or any(
            ra[0] != rb[0] or any(abs(x - y) > tol for x, y in zip(ra[1:], rb[1:]))
            for ra, rb in zip(a, b)
        ):
            diffs.append(f"labels differ: {rel}")
    return diffs


def _parse(path: Path) -> list[tuple]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            parts = line.split()
            rows.append((int(parts[0]), *map(float, parts[1:])))
    return rows


def load_coco_instances(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
