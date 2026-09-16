"""
Integrity checks on the local navigation dataset (data/nav_combined).

`data/` is gitignored, so every test skips on a fresh clone. When the dataset
is present these scan every file: a single bad label is enough to corrupt a
training run.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nav_dataset import CLASS_NAMES, SPLITS, read_manifest  # noqa: E402

EXPECTED_IMAGES = {"train": 8000 + 3608, "val": 2000 + 571}


@pytest.fixture(scope="module")
def dataset_dir():
    path = PROJECT_ROOT / "data" / "nav_combined"
    if not (path / "images").exists():
        pytest.skip("data/nav_combined not found (build it with scripts/build_dataset.py)")
    return path


@pytest.mark.parametrize("split", SPLITS)
def test_images_and_labels_pair_up_exactly(dataset_dir, split):
    images = {p.stem for p in (dataset_dir / "images" / split).iterdir()}
    labels = {p.stem for p in (dataset_dir / "labels" / split).glob("*.txt")}
    assert len(images) == EXPECTED_IMAGES[split]
    assert images == labels


@pytest.mark.parametrize("split", SPLITS)
def test_images_are_exactly_the_manifest(dataset_dir, split):
    manifest = {f"coco_{int(i):012d}" for i in read_manifest(PROJECT_ROOT / "data_manifest" / f"coco_{split}.txt")}
    manifest |= {f"custom_{i}" for i in read_manifest(PROJECT_ROOT / "data_manifest" / f"openimages_{split}.txt")}
    assert {p.stem for p in (dataset_dir / "images" / split).iterdir()} == manifest


@pytest.mark.parametrize("split", SPLITS)
def test_every_label_line_is_valid_yolo(dataset_dir, split):
    bad = []
    for label in (dataset_dir / "labels" / split).glob("*.txt"):
        lines = [line for line in label.read_text().splitlines() if line.strip()]
        if not lines:
            bad.append(f"{label.name}: empty")
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                bad.append(f"{label.name}: {line}")
                continue
            cls, coords = int(parts[0]), [float(x) for x in parts[1:]]
            if not 0 <= cls < len(CLASS_NAMES):
                bad.append(f"{label.name}: class {cls}")
            if not all(-1e-6 <= v <= 1 + 1e-6 for v in coords) or coords[2] <= 0 or coords[3] <= 0:
                bad.append(f"{label.name}: coords {coords}")
    assert bad == [], bad[:10]


def test_coco_images_only_use_coco_ids_and_openimages_only_custom_ids(dataset_dir):
    wrong = []
    for label in (dataset_dir / "labels" / "val").glob("*.txt"):
        ids = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
        if label.stem.startswith("coco_") and max(ids) >= 18:
            wrong.append(label.name)
        if label.stem.startswith("custom_") and min(ids) < 18:
            wrong.append(label.name)
    assert wrong == []


def test_every_class_appears_in_validation(dataset_dir):
    seen = set()
    for label in (dataset_dir / "labels" / "val").glob("*.txt"):
        seen |= {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}
    assert seen == set(range(len(CLASS_NAMES)))
