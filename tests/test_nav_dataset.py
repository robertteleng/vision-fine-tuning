"""Tests for src/nav_dataset.py: label conversion, dataset writing and manifests."""

import csv
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nav_dataset import (  # noqa: E402
    CLASS_NAMES,
    COCO_CLASSES,
    OPENIMAGES_CLASSES,
    Sample,
    coco_labels,
    compare_labels,
    dataset_stats,
    format_yolo_line,
    openimages_class_names,
    openimages_labels,
    read_manifest,
    write_dataset,
)

MANIFEST_DIR = PROJECT_ROOT / "data_manifest"


def parse(line):
    parts = line.split()
    return int(parts[0]), [float(x) for x in parts[1:]]


# --------------------------------------------------------------------------- classes


def test_class_layout_is_18_coco_then_6_openimages():
    assert len(COCO_CLASSES) == 18
    assert CLASS_NAMES[:18] == COCO_CLASSES
    assert CLASS_NAMES[18:] == OPENIMAGES_CLASSES
    assert CLASS_NAMES.index("Stairs") == 19
    assert len(set(CLASS_NAMES)) == 24


def test_class_order_matches_the_trained_model():
    weights = PROJECT_ROOT / "models" / "yolo26s_nav.pt"
    if not weights.exists():
        pytest.skip("models/yolo26s_nav.pt not present")
    from ultralytics import YOLO

    assert tuple(YOLO(str(weights)).names.values()) == CLASS_NAMES


# --------------------------------------------------------------------------- manifests


def test_read_manifest_ignores_blank_lines(tmp_path):
    path = tmp_path / "m.txt"
    path.write_text("30\n\n36\n  42  \n")
    assert read_manifest(path) == ["30", "36", "42"]


def test_read_manifest_rejects_duplicates(tmp_path):
    path = tmp_path / "m.txt"
    path.write_text("30\n36\n30\n")
    with pytest.raises(ValueError, match="duplicated"):
        read_manifest(path)


@pytest.mark.parametrize("source,train,val", [("coco", 8000, 2000), ("openimages", 3608, 571)])
def test_repo_manifests_have_expected_sizes_and_no_train_val_overlap(source, train, val):
    train_ids = read_manifest(MANIFEST_DIR / f"{source}_train.txt")
    val_ids = read_manifest(MANIFEST_DIR / f"{source}_val.txt")
    assert (len(train_ids), len(val_ids)) == (train, val)
    assert not set(train_ids) & set(val_ids)


# --------------------------------------------------------------------------- COCO


def coco_fixture():
    return {
        "categories": [{"id": 1, "name": "person"}, {"id": 5, "name": "airplane"}, {"id": 18, "name": "dog"}],
        "images": [
            {"id": 7, "width": 200, "height": 100},
            {"id": 8, "width": 50, "height": 50},
            {"id": 9, "width": 10, "height": 10},
        ],
        "annotations": [
            {"image_id": 7, "category_id": 1, "bbox": [20, 10, 40, 30], "iscrowd": 0},
            {"image_id": 7, "category_id": 5, "bbox": [0, 0, 10, 10], "iscrowd": 0},  # not a nav class
            {"image_id": 7, "category_id": 18, "bbox": [100, 50, 100, 50], "iscrowd": 1},  # crowd kept
            {"image_id": 8, "category_id": 5, "bbox": [0, 0, 5, 5], "iscrowd": 0},
            {"image_id": 9, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},  # not requested
        ],
    }


def test_coco_labels_filters_classes_and_converts_boxes():
    lines = coco_labels(coco_fixture(), [7])
    assert list(lines) == [7]
    rows = sorted(parse(line) for line in lines[7])
    assert rows[0] == (COCO_CLASSES.index("person"), pytest.approx([0.2, 0.25, 0.2, 0.3]))
    assert rows[1] == (COCO_CLASSES.index("dog"), pytest.approx([0.75, 0.75, 0.5, 0.5]))
    assert len(rows) == 2  # the airplane is dropped


def test_coco_labels_rejects_unknown_image():
    with pytest.raises(KeyError, match="not in annotations"):
        coco_labels(coco_fixture(), [7, 12345])


def test_coco_labels_rejects_image_without_nav_classes():
    with pytest.raises(ValueError, match="without target classes"):
        coco_labels(coco_fixture(), [8])


# --------------------------------------------------------------------------- Open Images


OI_HEADER = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax",
             "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]


def write_openimages_fixture(tmp_path):
    classes = tmp_path / "classes.csv"
    classes.write_text("/m/door,Door\n/m/stairs,Stairs\n/m/wheel,Wheelchair\n/m/car,Car\n")
    detections = tmp_path / "detections.csv"
    with open(detections, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OI_HEADER)
        w.writerow(["aaa", "xclick", "/m/stairs", 1, 0.1, 0.5, 0.2, 0.6, 0, 0, 0, 0, 0])
        w.writerow(["aaa", "xclick", "/m/car", 1, 0.0, 0.1, 0.0, 0.1, 0, 0, 0, 0, 0])  # COCO-like class, dropped
        w.writerow(["aaa", "xclick", "/m/wheel", 1, 0.0, 1.0, 0.0, 1.0, 0, 0, 1, 0, 0])  # group box kept
        w.writerow(["bbb", "xclick", "/m/door", 1, 0.2, 0.4, 0.2, 0.4, 0, 0, 0, 0, 0])  # not requested
        w.writerow(["ccc", "xclick", "/m/car", 1, 0.2, 0.4, 0.2, 0.4, 0, 0, 0, 0, 0])
    return classes, detections


def test_openimages_labels_offsets_ids_after_coco(tmp_path):
    classes, detections = write_openimages_fixture(tmp_path)
    lines = openimages_labels(detections, openimages_class_names(classes), ["aaa"])
    rows = sorted(parse(line) for line in lines["aaa"])
    assert rows[0] == (19, pytest.approx([0.3, 0.4, 0.4, 0.4]))  # Stairs
    assert rows[1] == (23, pytest.approx([0.5, 0.5, 1.0, 1.0]))  # Wheelchair, IsGroupOf
    assert len(rows) == 2


def test_openimages_labels_rejects_image_without_target_classes(tmp_path):
    classes, detections = write_openimages_fixture(tmp_path)
    with pytest.raises(ValueError, match="without target classes"):
        openimages_labels(detections, openimages_class_names(classes), ["ccc"])


def test_format_yolo_line_uses_six_decimals():
    assert format_yolo_line(3, 0.5, 1 / 3, 0.25, 2 / 3) == "3 0.500000 0.333333 0.250000 0.666667"


# --------------------------------------------------------------------------- writing


def make_samples(tmp_path):
    img = tmp_path / "src.jpg"
    img.write_bytes(b"\xff\xd8fake")
    return [
        Sample("train", "coco_000000000007", img, ("0 0.5 0.5 0.2 0.2", "11 0.1 0.1 0.1 0.1")),
        Sample("val", "custom_aaa", img, ("19 0.3 0.4 0.4 0.4",)),
    ]


@pytest.mark.parametrize("link", ["hard", "symlink", "copy"])
def test_write_dataset_layout_and_yaml(tmp_path, link):
    out = tmp_path / "ds"
    write_dataset(make_samples(tmp_path), out, link=link)

    assert (out / "images/train/coco_000000000007.jpg").read_bytes() == b"\xff\xd8fake"
    assert (out / "labels/val/custom_aaa.txt").read_text() == "19 0.3 0.4 0.4 0.4\n"
    if link == "symlink":
        assert (out / "images/val/custom_aaa.jpg").is_symlink()

    cfg = yaml.safe_load((out / "dataset.yaml").read_text())
    assert Path(cfg["path"]).is_absolute()
    assert (cfg["train"], cfg["val"]) == ("images/train", "images/val")
    assert tuple(cfg["names"][i] for i in range(24)) == CLASS_NAMES


def test_write_dataset_rejects_unknown_link_mode(tmp_path):
    with pytest.raises(ValueError, match="unknown link mode"):
        write_dataset(make_samples(tmp_path), tmp_path / "ds", link="teleport")


def test_dataset_stats_counts_instances_and_images_per_class(tmp_path):
    samples = make_samples(tmp_path) + [Sample("train", "coco_2", tmp_path / "src.jpg", ("0 0.1 0.1 0.1 0.1",) * 3)]
    stats = dataset_stats(samples)
    assert stats["train"]["images"] == 2
    assert stats["train"]["classes"]["person"] == {"instances": 4, "images": 2}
    assert stats["train"]["classes"]["dog"] == {"instances": 1, "images": 1}
    assert stats["val"]["classes"]["Stairs"] == {"instances": 1, "images": 1}
    assert stats["val"]["classes"]["Door"] == {"instances": 0, "images": 0}


# --------------------------------------------------------------------------- verification


def write_labels(root, name, text):
    path = root / "train" / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_compare_labels_accepts_rounding_and_order(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_labels(a, "x", "0 0.500000 0.500000 0.2 0.2\n19 0.1 0.1 0.1 0.1\n")
    write_labels(b, "x", "19 0.10004 0.1 0.1 0.1\n0 0.5 0.5 0.2 0.2\n")
    assert compare_labels(a, b) == []


def test_compare_labels_reports_every_kind_of_difference(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_labels(a, "only_a", "0 0.5 0.5 0.2 0.2\n")
    write_labels(b, "only_b", "0 0.5 0.5 0.2 0.2\n")
    write_labels(a, "cls", "0 0.5 0.5 0.2 0.2\n")
    write_labels(b, "cls", "1 0.5 0.5 0.2 0.2\n")
    write_labels(a, "coord", "0 0.5 0.5 0.2 0.2\n")
    write_labels(b, "coord", "0 0.51 0.5 0.2 0.2\n")
    write_labels(a, "count", "0 0.5 0.5 0.2 0.2\n")
    write_labels(b, "count", "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
    diffs = compare_labels(a, b)
    assert "missing: train/only_a.txt" in diffs
    assert "extra: train/only_b.txt" in diffs
    assert {"labels differ: train/cls.txt", "labels differ: train/coord.txt",
            "labels differ: train/count.txt"} <= set(diffs)
