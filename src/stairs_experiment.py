"""Stairs auto-annotation experiment (docs/EXPERIMENT_STAIRS.md).

Selection of the extra and test images, YOLO labels for a single class, and
the quality of pseudo-labels against human boxes. GPU-free and unit-tested;
Grounding DINO and training are driven by scripts/experiment_stairs.py.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from src.nav_dataset import CLASS_NAMES, format_yolo_line

STAIRS_ID = CLASS_NAMES.index("Stairs")

Box = tuple[float, float, float, float]  # normalized x0, y0, x1, y1


def stairs_boxes(detections_csv: Path, stairs_mid: str) -> dict[str, list[tuple[Box, bool]]]:
    """All `Stairs` boxes per image as (box, is_group_of), streamed from an Open Images CSV."""
    boxes: dict[str, list[tuple[Box, bool]]] = defaultdict(list)
    with open(detections_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["LabelName"] != stairs_mid:
                continue
            box = (float(row["XMin"]), float(row["YMin"]), float(row["XMax"]), float(row["YMax"]))
            boxes[row["ImageID"]].append((box, row["IsGroupOf"] == "1"))
    return dict(boxes)


def select_images(boxes: dict[str, list[tuple[Box, bool]]], exclude: Iterable[str], n: int) -> list[str]:
    """First `n` image IDs (sorted) with at least one non-group box, skipping `exclude`."""
    excluded = set(exclude)
    eligible = sorted(i for i, bs in boxes.items() if i not in excluded and any(not g for _, g in bs))
    return eligible[:n]


def yolo_lines(boxes: Sequence[Box], class_id: int = STAIRS_ID) -> list[str]:
    return [format_yolo_line(class_id, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0) for x0, y0, x1, y1 in boxes]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_counts(predicted: Sequence[Box], truth: Sequence[Box], threshold: float = 0.5) -> tuple[int, int, int]:
    """Greedy one-to-one matching by IoU. Returns (true positives, false positives, false negatives)."""
    pairs = sorted(((iou(p, t), pi, ti) for pi, p in enumerate(predicted) for ti, t in enumerate(truth)), reverse=True)
    used_p, used_t = set(), set()
    for score, pi, ti in pairs:
        if score < threshold:
            break
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
    tp = len(used_p)
    return tp, len(predicted) - tp, len(truth) - tp


def pseudo_label_quality(predicted: dict[str, Sequence[Box]], truth: dict[str, Sequence[Box]],
                         threshold: float = 0.5) -> dict:
    """Precision and recall of pseudo-labels over a set of images (missing entries = no boxes)."""
    tp = fp = fn = 0
    empty_images = 0
    for image_id in truth.keys() | predicted.keys():
        p, t = predicted.get(image_id, []), truth.get(image_id, [])
        empty_images += not p
        a, b, c = match_counts(p, t, threshold)
        tp, fp, fn = tp + a, fp + b, fn + c
    return {
        "iou_threshold": threshold,
        "images": len(truth.keys() | predicted.keys()),
        "images_without_pseudo_labels": empty_images,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }
