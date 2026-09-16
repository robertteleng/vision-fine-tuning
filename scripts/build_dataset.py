#!/usr/bin/env python3
"""
Build the navigation dataset (step 1 of the pipeline).

Images come from the FiftyOne zoo cache; anything missing is downloaded by ID
with --download. Labels are converted from the original COCO and Open Images
annotation files. The image selection is fixed by data_manifest/.

Usage:
    uv sync --extra datasets
    uv run python scripts/build_dataset.py --out data/nav_combined --download
    uv run python scripts/build_dataset.py --out data/nav_rebuilt --verify data/nav_combined
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nav_dataset import (
    COCO_PREFIX,
    FIFTYONE_SPLIT,
    OPENIMAGES_PREFIX,
    SPLITS,
    Sample,
    coco_labels,
    compare_labels,
    dataset_stats,
    load_coco_instances,
    missing_images,
    openimages_class_names,
    openimages_labels,
    read_manifest,
    write_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MANIFEST_DIR = PROJECT_ROOT / "data_manifest"


def coco_samples(fiftyone_dir: Path, split: str, ids: list[str]) -> list[Sample]:
    coco_dir = fiftyone_dir / "coco-2017"
    instances = load_coco_instances(coco_dir / "raw" / f"instances_{split}2017.json")
    lines = coco_labels(instances, [int(i) for i in ids])
    return [
        Sample(split, f"{COCO_PREFIX}{int(i):012d}",
               coco_dir / FIFTYONE_SPLIT[split] / "data" / f"{int(i):012d}.jpg", tuple(lines[int(i)]))
        for i in ids
    ]


def openimages_samples(fiftyone_dir: Path, split: str, ids: list[str]) -> list[Sample]:
    oi_dir = fiftyone_dir / "open-images-v7"
    names = openimages_class_names(oi_dir / "train" / "metadata" / "classes.csv")
    lines = openimages_labels(oi_dir / FIFTYONE_SPLIT[split] / "labels" / "detections.csv", names, ids)
    return [
        Sample(split, f"{OPENIMAGES_PREFIX}{i}", oi_dir / FIFTYONE_SPLIT[split] / "data" / f"{i}.jpg", tuple(lines[i]))
        for i in ids
    ]


def download(fiftyone_dir: Path, source: str, split: str, ids: list[str]) -> None:
    """Fetch only the listed images (and the annotation files) into the FiftyOne cache."""
    if source == "coco":
        import fiftyone.utils.coco as fouc

        fouc.download_coco_dataset_split(
            str(fiftyone_dir / "coco-2017" / FIFTYONE_SPLIT[split]), FIFTYONE_SPLIT[split], year="2017",
            label_types=["detections"], image_ids=[int(i) for i in ids],
            raw_dir=str(fiftyone_dir / "coco-2017" / "raw"),
        )
    else:
        import fiftyone.utils.openimages as fouo

        fouo.download_open_images_split(
            str(fiftyone_dir / "open-images-v7" / FIFTYONE_SPLIT[split]), FIFTYONE_SPLIT[split],
            version="v7", label_types=["detections"], image_ids=ids,
        )


def main():
    parser = argparse.ArgumentParser(description="Navigation detector — build dataset")
    parser.add_argument("--out", type=Path, required=True, help="Output dataset directory")
    parser.add_argument("--fiftyone-dir", type=Path, default=Path.home() / "fiftyone", help="FiftyOne zoo cache")
    parser.add_argument("--download", action="store_true", help="Download images missing from the cache")
    parser.add_argument("--link", choices=["hard", "symlink", "copy"], default="hard", help="How to place images")
    parser.add_argument("--verify", type=Path, default=None, help="Existing dataset to compare labels against")
    args = parser.parse_args()

    samples: list[Sample] = []
    for source, build in (("coco", coco_samples), ("openimages", openimages_samples)):
        for split in SPLITS:
            ids = read_manifest(MANIFEST_DIR / f"{source}_{split}.txt")
            if args.download:
                download(args.fiftyone_dir, source, split, ids)
            logger.info(f"{source}/{split}: {len(ids)} images")
            samples += build(args.fiftyone_dir, split, ids)

    missing = missing_images(samples)
    if missing:
        logger.error(f"{len(missing)} images missing from the cache (e.g. {missing[0]}). Re-run with --download.")
        sys.exit(1)

    write_dataset(samples, args.out, link=args.link)
    stats = dataset_stats(samples)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    for split in SPLITS:
        logger.info(f"{split}: {stats[split]['images']} images")
    logger.info(f"Dataset written to {args.out}")

    if args.verify:
        diffs = compare_labels(args.verify / "labels", args.out / "labels")
        if diffs:
            logger.error(f"{len(diffs)} differences against {args.verify}, e.g. {diffs[:5]}")
            sys.exit(1)
        logger.info(f"Labels identical to {args.verify} (tolerance 1e-4)")


if __name__ == "__main__":
    main()
