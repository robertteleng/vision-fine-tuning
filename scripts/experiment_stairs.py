#!/usr/bin/env python3
"""
Stairs auto-annotation experiment — runs the pre-registered design in
docs/EXPERIMENT_STAIRS.md, one step at a time.

Usage:
    uv sync --extra datasets --extra annotation
    uv run python scripts/experiment_stairs.py select        # pin extra + test images, download them
    uv run python scripts/experiment_stairs.py pseudo-label  # Grounding DINO on the extra images (GPU)
    uv run python scripts/experiment_stairs.py quality       # pseudo-labels vs human boxes
    uv run python scripts/experiment_stairs.py build         # datasets for arms B and C + test set
    uv run python scripts/experiment_stairs.py train --arm B # then --arm C (GPU)
    uv run python scripts/experiment_stairs.py evaluate      # A, B, C on test set and val (GPU)
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.edge_bench import EVAL_SETTINGS, accuracy_summary  # noqa: E402
from src.nav_dataset import CLASS_NAMES, read_manifest, write_dataset_yaml  # noqa: E402
from src.stairs_experiment import (  # noqa: E402
    pseudo_label_quality,
    select_images,
    stairs_boxes,
    yolo_lines,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Fixed in docs/EXPERIMENT_STAIRS.md before running anything.
N_EXTRA, N_TEST = 1000, 500
GDINO_MODEL, GDINO_PROMPT, GDINO_THRESHOLD = "IDEA-Research/grounding-dino-tiny", "stairs.", 0.30

OI = Path.home() / "fiftyone" / "open-images-v7"
MANIFESTS = PROJECT_ROOT / "data_manifest"
EXP = PROJECT_ROOT / "data" / "experiment_stairs"
ARM_A_WEIGHTS = PROJECT_ROOT / "models" / "yolo26n_nav.pt"
RESULTS = PROJECT_ROOT / "benchmarks" / "experiment_stairs"


def stairs_mid() -> str:
    for row in csv.reader(open(OI / "train" / "metadata" / "classes.csv")):
        if row[1] == "Stairs":
            return row[0]
    raise KeyError("Stairs not in Open Images classes")


def download(split: str, **kwargs) -> None:
    import fiftyone.utils.openimages as fouo

    fouo.download_open_images_split(str(OI / split), split, version="v7", label_types=["detections"], **kwargs)


def cmd_select(_):
    mid = stairs_mid()
    used = set(read_manifest(MANIFESTS / "openimages_train.txt")) | set(read_manifest(MANIFESTS / "openimages_val.txt"))
    extra = select_images(stairs_boxes(OI / "train" / "labels" / "detections.csv", mid), used, N_EXTRA)

    if not (OI / "test" / "labels" / "detections.csv").exists():
        download("test", classes=["Stairs"], max_samples=1)  # fetches the test annotations
    test = select_images(stairs_boxes(OI / "test" / "labels" / "detections.csv", mid), set(), N_TEST)

    (MANIFESTS / "experiment_stairs_train.txt").write_text("\n".join(extra) + "\n")
    (MANIFESTS / "experiment_stairs_test.txt").write_text("\n".join(test) + "\n")
    logger.info(f"pinned {len(extra)} extra train images and {len(test)} test images")
    download("train", image_ids=extra)
    download("test", image_ids=test)


def human_boxes(split: str, ids) -> dict:
    boxes = stairs_boxes(OI / split / "labels" / "detections.csv", stairs_mid())
    return {i: [b for b, _ in boxes[i]] for i in ids}


def cmd_pseudo_label(_):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    ids = read_manifest(MANIFESTS / "experiment_stairs_train.txt")
    device = "cuda" if torch.cuda.is_available() else sys.exit("Grounding DINO needs the GPU")
    processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(device).eval()

    out = {}
    for n, image_id in enumerate(ids, 1):
        image = Image.open(OI / "train" / "data" / f"{image_id}.jpg").convert("RGB")
        w, h = image.size
        inputs = processor(images=image, text=GDINO_PROMPT, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs.input_ids, target_sizes=[(h, w)])[0]
        keep = result["scores"] >= GDINO_THRESHOLD
        out[image_id] = [
            {"box": [float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h], "score": float(s)}
            for (x0, y0, x1, y1), s in zip(result["boxes"][keep].tolist(), result["scores"][keep].tolist())
        ]
        if n % 100 == 0:
            logger.info(f"{n}/{len(ids)}")
    EXP.mkdir(parents=True, exist_ok=True)
    meta = {"model": GDINO_MODEL, "prompt": GDINO_PROMPT, "threshold": GDINO_THRESHOLD, "images": len(ids)}
    (EXP / "pseudo_labels.json").write_text(json.dumps({"meta": meta, "boxes": out}, indent=1))
    logger.info(f"pseudo-labels for {len(out)} images, {sum(map(len, out.values()))} boxes")


def load_pseudo() -> dict:
    data = json.loads((EXP / "pseudo_labels.json").read_text())
    return {i: [tuple(b["box"]) for b in bs] for i, bs in data["boxes"].items()}


def cmd_quality(_):
    ids = read_manifest(MANIFESTS / "experiment_stairs_train.txt")
    q = pseudo_label_quality(load_pseudo(), human_boxes("train", ids))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pseudo_label_quality.json").write_text(json.dumps(q, indent=2) + "\n")
    logger.info(json.dumps(q, indent=2))


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_arm(name: str, extra_labels: dict) -> Path:
    base = PROJECT_ROOT / "data" / "nav_combined"
    root = EXP / f"arm_{name}"
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        for img in (base / "images" / split).iterdir():
            link(img, root / "images" / split / img.name)
            link(base / "labels" / split / f"{img.stem}.txt", root / "labels" / split / f"{img.stem}.txt")
    for image_id, boxes in extra_labels.items():
        link(OI / "train" / "data" / f"{image_id}.jpg", root / "images" / "train" / f"stairsx_{image_id}.jpg")
        (root / "labels" / "train" / f"stairsx_{image_id}.txt").write_text("\n".join(yolo_lines(boxes)) + "\n")
    write_dataset_yaml(root)
    return root


def cmd_build(_):
    extra = read_manifest(MANIFESTS / "experiment_stairs_train.txt")
    build_arm("B", load_pseudo())
    build_arm("C", human_boxes("train", extra))

    test_ids = read_manifest(MANIFESTS / "experiment_stairs_test.txt")
    test_root = EXP / "test"
    if test_root.exists():
        shutil.rmtree(test_root)
    for image_id, boxes in human_boxes("test", test_ids).items():
        link(OI / "test" / "data" / f"{image_id}.jpg", test_root / "images" / "val" / f"{image_id}.jpg")
        (test_root / "labels" / "val" / f"{image_id}.txt").parent.mkdir(parents=True, exist_ok=True)
        (test_root / "labels" / "val" / f"{image_id}.txt").write_text("\n".join(yolo_lines(boxes)) + "\n")
    (test_root / "images" / "train").mkdir(parents=True, exist_ok=True)  # Ultralytics wants the key to exist
    write_dataset_yaml(test_root)
    logger.info("built arm_B, arm_C and test")


def cmd_train(args):
    from ultralytics import YOLO
    import yaml

    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    keys = ("epochs", "patience", "batch", "imgsz", "workers", "cache", "lr0", "lrf", "momentum", "weight_decay",
            "optimizer", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "close_mosaic", "cos_lr", "fliplr",
            "flipud", "degrees", "translate", "scale", "shear", "perspective", "hsv_h", "hsv_s", "hsv_v", "mosaic",
            "mixup", "copy_paste", "device", "amp", "deterministic", "seed")
    params = {k: cfg[k] for k in keys if k in cfg}
    YOLO("yolo26n.pt").train(data=str(EXP / f"arm_{args.arm}" / "dataset.yaml"),
                             project=str(PROJECT_ROOT / "runs" / "experiment_stairs"), name=f"arm_{args.arm}",
                             exist_ok=True, **params)


def cmd_evaluate(_):
    from ultralytics import YOLO

    arms = {"A": ARM_A_WEIGHTS}
    for arm in ("B", "C"):
        arms[arm] = PROJECT_ROOT / "runs" / "experiment_stairs" / f"arm_{arm}" / "weights" / "best.pt"
    results = {}
    for arm, weights in arms.items():
        model = YOLO(str(weights))
        scratch = str(PROJECT_ROOT / "runs" / "experiment_stairs" / "eval")
        test = accuracy_summary(model.val(data=str(EXP / "test" / "dataset.yaml"), split="val", device=0, plots=False,
                                          verbose=False, project=scratch, name=f"test_{arm}", exist_ok=True,
                                          **EVAL_SETTINGS), model.names)
        val = accuracy_summary(model.val(data=str(PROJECT_ROOT / "data/nav_combined/dataset.yaml"), split="val",
                                         device=0, plots=False, verbose=False, project=scratch, name=f"val_{arm}",
                                         exist_ok=True, **EVAL_SETTINGS), model.names)
        results[arm] = {"weights": str(weights), "stairs_test_ap50": test["per_class"]["Stairs"]["map50"],
                        "stairs_test_ap50_95": test["per_class"]["Stairs"]["map50_95"], "val": val}
        logger.info(f"arm {arm}: Stairs test AP50 {results[arm]['stairs_test_ap50']:.3f} | val mAP50 {val['map50']:.3f}")

    gain = {arm: results[arm]["stairs_test_ap50"] - results["A"]["stairs_test_ap50"] for arm in ("B", "C")}
    val_drop_b = results["A"]["val"]["map50"] - results["B"]["val"]["map50"]
    if gain["C"] < 0.02:
        verdict = "more data of this kind does not help Stairs; no verdict on auto-annotation"
    elif gain["B"] < 0:
        verdict = "pseudo-labels hurt"
    elif gain["B"] >= 0.5 * gain["C"] and val_drop_b <= 0.01:
        verdict = "auto-annotation is worth using for Stairs"
    else:
        verdict = "auto-annotation captures less than half of the human-label gain (or regresses val)"
    summary = {"arms": results, "gain": gain, "val_map50_drop_B": val_drop_b, "verdict": verdict,
               "criterion": "docs/EXPERIMENT_STAIRS.md (pre-registered 2026-09-16)"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info(f"gain B {gain['B']:+.3f} | gain C {gain['C']:+.3f} | val drop B {val_drop_b:+.3f} -> {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Stairs auto-annotation experiment")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("select", "pseudo-label", "quality", "build", "evaluate"):
        sub.add_parser(name)
    train = sub.add_parser("train")
    train.add_argument("--arm", choices=["B", "C"], required=True)
    args = parser.parse_args()
    {"select": cmd_select, "pseudo-label": cmd_pseudo_label, "quality": cmd_quality, "build": cmd_build,
     "train": cmd_train, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
