"""Tests for src/stairs_experiment.py."""

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stairs_experiment import (  # noqa: E402
    STAIRS_ID,
    iou,
    match_counts,
    pseudo_label_quality,
    select_images,
    stairs_boxes,
    yolo_lines,
)

HEADER = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax",
          "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def test_stairs_id_is_19():
    assert STAIRS_ID == 19


def test_stairs_boxes_keeps_only_stairs_with_group_flag(tmp_path):
    path = tmp_path / "d.csv"
    write_csv(path, [
        ["b", "xclick", "/m/stairs", 1, 0.1, 0.5, 0.2, 0.6, 0, 0, 0, 0, 0],
        ["b", "xclick", "/m/stairs", 1, 0.0, 1.0, 0.0, 1.0, 0, 0, 1, 0, 0],
        ["b", "xclick", "/m/door", 1, 0.1, 0.2, 0.1, 0.2, 0, 0, 0, 0, 0],
        ["a", "xclick", "/m/door", 1, 0.1, 0.2, 0.1, 0.2, 0, 0, 0, 0, 0],
    ])
    boxes = stairs_boxes(path, "/m/stairs")
    assert list(boxes) == ["b"]
    assert boxes["b"] == [((0.1, 0.2, 0.5, 0.6), False), ((0.0, 0.0, 1.0, 1.0), True)]


def test_select_images_sorted_skips_excluded_and_group_only():
    boxes = {
        "d": [((0, 0, 1, 1), False)],
        "a": [((0, 0, 1, 1), True)],  # group only -> not eligible
        "c": [((0, 0, 1, 1), True), ((0, 0, .5, .5), False)],
        "b": [((0, 0, 1, 1), False)],  # excluded
        "e": [((0, 0, 1, 1), False)],
    }
    assert select_images(boxes, exclude={"b"}, n=2) == ["c", "d"]
    assert select_images(boxes, exclude=set(), n=10) == ["b", "c", "d", "e"]


def test_yolo_lines_converts_corners_to_center_format():
    assert yolo_lines([(0.1, 0.2, 0.5, 0.6)]) == ["19 0.300000 0.400000 0.400000 0.400000"]


def test_iou_cases():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == pytest.approx(1.0)
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0
    assert iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_match_counts_is_one_to_one():
    truth = [(0, 0, 1, 1)]
    predicted = [(0, 0, 1, 1), (0.01, 0, 1, 1)]  # duplicate detection of the same object
    assert match_counts(predicted, truth) == (1, 1, 0)
    assert match_counts([], truth) == (0, 0, 1)
    assert match_counts(predicted, []) == (0, 2, 0)


def test_match_counts_respects_threshold():
    assert match_counts([(0, 0, 2, 2)], [(1, 1, 3, 3)], threshold=0.5) == (0, 1, 1)
    assert match_counts([(0, 0, 2, 2)], [(1, 1, 3, 3)], threshold=0.1) == (1, 0, 0)


def test_pseudo_label_quality_aggregates_over_images():
    truth = {"a": [(0, 0, 1, 1)], "b": [(0, 0, 1, 1), (2, 2, 3, 3)]}
    predicted = {"a": [(0, 0, 1, 1)], "b": [(2, 2, 3, 3), (5, 5, 6, 6)], "c": [(0, 0, 1, 1)]}
    q = pseudo_label_quality(predicted, truth)
    assert (q["true_positives"], q["false_positives"], q["false_negatives"]) == (2, 2, 1)
    assert q["precision"] == pytest.approx(0.5)
    assert q["recall"] == pytest.approx(2 / 3)
    assert q["images"] == 3


def test_pseudo_label_quality_counts_images_left_without_labels():
    q = pseudo_label_quality({}, {"a": [(0, 0, 1, 1)]})
    assert q["images_without_pseudo_labels"] == 1
    assert (q["precision"], q["recall"]) == (0.0, 0.0)
