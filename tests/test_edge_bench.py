"""Tests for src/edge_bench.py: the parts of the benchmark protocol that must not drift."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import edge_bench as eb  # noqa: E402

# --------------------------------------------------------------------------- latency


def test_latency_summary_percentiles_and_fps():
    s = eb.latency_summary(list(range(1, 101)))  # 1..100 ms
    assert s["n"] == 100
    assert s["mean"] == pytest.approx(50.5)
    assert s["p50"] == pytest.approx(50.5)
    assert s["p95"] == pytest.approx(95.05)
    assert s["p99"] == pytest.approx(99.01)
    assert (s["min"], s["max"]) == (1, 100)
    assert s["fps"] == pytest.approx(1000 / 50.5)


def test_latency_summary_tail_is_visible_when_mean_hides_it():
    s = eb.latency_summary([2.0] * 95 + [40.0] * 5)
    assert s["p50"] == 2.0
    assert s["p99"] == 40.0
    assert s["mean"] < 5


def test_latency_summary_rejects_empty():
    with pytest.raises(ValueError):
        eb.latency_summary([])


# --------------------------------------------------------------------------- sampling and calibration


def make_dataset(tmp_path, n_train=20, n_val=5):
    for split, n in (("train", n_train), ("val", n_val)):
        d = tmp_path / "ds" / "images" / split
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{split}_{i:03d}.jpg").write_bytes(b"x")
        (d / "notes.txt").write_text("not an image")
    cfg = {"path": str(tmp_path / "ds"), "train": "images/train", "val": "images/val", "names": {0: "person", 1: "Stairs"}}
    path = tmp_path / "ds" / "dataset.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_list_images_is_sorted_and_skips_non_images(tmp_path):
    make_dataset(tmp_path)
    images = eb.list_images(tmp_path / "ds/images/train")
    assert len(images) == 20
    assert images == sorted(images)


def test_sample_images_is_deterministic_and_seed_dependent(tmp_path):
    images = [tmp_path / f"{i}.jpg" for i in range(100)]
    a = eb.sample_images(list(reversed(images)), 10, seed=0)
    assert a == eb.sample_images(images, 10, seed=0)  # input order does not matter
    assert a != eb.sample_images(images, 10, seed=1)
    assert eb.sample_images(images, 1000, seed=0) == sorted(images)


def test_calibration_yaml_uses_only_train_images(tmp_path):
    dataset_yaml = make_dataset(tmp_path)
    calib_yaml = eb.write_calibration_yaml(dataset_yaml, tmp_path / "calib", n=8, seed=3)
    cfg = yaml.safe_load(calib_yaml.read_text())
    listing = Path(cfg["path"]) / cfg["val"]
    paths = [Path(p) for p in listing.read_text().split()]

    assert len(paths) == 8
    assert all("/images/train/" in str(p) for p in paths)
    assert not any("/images/val/" in str(p) for p in paths)
    assert cfg["names"] == {0: "person", 1: "Stairs"}
    assert eb.write_calibration_yaml(dataset_yaml, tmp_path / "calib2", n=8, seed=3).read_text().replace(
        "calib2", "calib") == calib_yaml.read_text()


def test_dataset_split_dir(tmp_path):
    dataset_yaml = make_dataset(tmp_path)
    assert eb.dataset_split_dir(dataset_yaml, "val") == tmp_path / "ds/images/val"


# --------------------------------------------------------------------------- artifacts


def test_engine_path_includes_precision_and_host(tmp_path):
    p = eb.engine_path(Path("models/yolo26n_nav.pt"), "int8", "jetson", tmp_path)
    assert p == tmp_path / "yolo26n_nav_int8_jetson.engine"


def test_engine_path_rejects_unknown_precision(tmp_path):
    with pytest.raises(ValueError):
        eb.engine_path(Path("m.pt"), "int4", "nuc", tmp_path)


def test_sha256(tmp_path):
    f = tmp_path / "f"
    f.write_bytes(b"abc")
    assert eb.sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# --------------------------------------------------------------------------- accuracy


def test_accuracy_summary_maps_class_indices_to_names():
    box = SimpleNamespace(mp=0.5, mr=0.4, map50=0.47, map=0.31,
                          ap50=np.array([0.7, 0.3]), ap=np.array([0.5, 0.2]), ap_class_index=np.array([0, 19]))
    acc = eb.accuracy_summary(SimpleNamespace(box=box), {0: "person", 19: "Stairs", 20: "Street light"})
    assert acc["map50"] == pytest.approx(0.47)
    assert acc["per_class"] == {"person": {"map50": 0.7, "map50_95": 0.5}, "Stairs": {"map50": 0.3, "map50_95": 0.2}}
    assert "Street light" not in acc["per_class"]


# --------------------------------------------------------------------------- records and tables


def fake_record(host="nuc", name="yolo26n_nav", precision="fp16", created="2026-09-16T20:00:00+00:00",
                p95=2.0, mean=1.8, map50=0.45, stairs=0.30, gpu="RTX"):
    return {
        "schema": eb.SCHEMA_VERSION,
        "created": created,
        "environment": {"host_tag": host, "gpu": gpu},
        "command": ["benchmark.py"],
        "model": {"name": name, "precision": precision, "artifact_mb": 7.5},
        "protocol": {},
        "latency_ms": {"end_to_end": {"mean": mean, "p95": p95, "fps": 1000 / mean}},
        "accuracy": {"map50": map50, "map50_95": 0.3, "per_class": {"Stairs": {"map50": stairs, "map50_95": 0.1}}},
        "memory": None,
    }


def test_write_and_load_records_round_trip(tmp_path):
    rec = fake_record()
    path = eb.write_record(rec, tmp_path)
    assert path.name == "20260916T200000_nuc_yolo26n_nav_fp16.json"
    (tmp_path / "old.json").write_text(json.dumps({"schema": 0}))
    assert eb.load_records(tmp_path) == [rec]


def test_latest_by_key_keeps_newest():
    old = fake_record(created="2026-09-16T10:00:00+00:00", map50=0.1)
    new = fake_record(created="2026-09-16T12:00:00+00:00", map50=0.2)
    assert eb.latest_by_key([new, old])[("nuc", "yolo26n_nav", "fp16")]["accuracy"]["map50"] == 0.2


def test_markdown_table_orders_precisions_and_marks_missing_classes():
    records = [fake_record(precision="int8"), fake_record(precision="fp32"), fake_record(precision="fp16")]
    table = eb.markdown_table(records, classes=("Stairs", "Door"))
    rows = table.strip().splitlines()[2:]
    assert [r.split("|")[3].strip() for r in rows] == ["FP32", "FP16", "INT8"]
    assert rows[0].rstrip(" |").endswith("0.300 | —")
    assert rows[0].split("|")[7].strip() == "—"  # no engine-only latency for this record


def test_markdown_table_shows_engine_only_latency():
    rec = fake_record()
    rec["latency_ms"]["engine_only"] = {"mean": 4.24}
    row = eb.markdown_table([rec], classes=()).strip().splitlines()[2]
    assert row.split("|")[7].strip() == "4.24"


# --------------------------------------------------------------------------- INT8 decision


def test_int8_verdict_passes_only_when_all_checks_pass():
    fp16 = fake_record(p95=2.0, map50=0.470, stairs=0.318)
    good = fake_record(precision="int8", p95=1.5, map50=0.465, stairs=0.305)
    assert eb.int8_verdict(fp16, good)["use_int8"] is True

    slow = fake_record(precision="int8", p95=1.9, map50=0.470, stairs=0.318)  # only 5% faster
    v = eb.int8_verdict(fp16, slow)
    assert v["use_int8"] is False
    assert v["checks"]["p95_latency_reduction"]["pass"] is False

    stairs_hit = fake_record(precision="int8", p95=1.0, map50=0.468, stairs=0.28)  # global fine, Stairs not
    v = eb.int8_verdict(fp16, stairs_hit)
    assert v["checks"]["map50_drop"]["pass"] is True
    assert v["checks"]["stairs_map50_drop"]["pass"] is False
    assert v["use_int8"] is False


def test_int8_criterion_was_fixed_before_measuring():
    assert eb.INT8_CRITERION["decided_on"] == "2026-09-16"
    assert eb.EVAL_SETTINGS == {"imgsz": 640, "batch": 1, "conf": 0.001, "iou": 0.7}


# --------------------------------------------------------------------------- environment


def test_environment_takes_commit_and_container_from_host_variables(monkeypatch):
    monkeypatch.setenv("BENCH_GIT_COMMIT", "abc123")
    monkeypatch.setenv("BENCH_GIT_DIRTY", "0")
    monkeypatch.setenv("BENCH_CONTAINER_IMAGE", "ultralytics/ultralytics:8.4.14-jetson-jetpack6")
    env = eb.environment()
    assert env["git_commit"] == "abc123"
    assert env["git_dirty"] is False
    assert env["container_image"] == "ultralytics/ultralytics:8.4.14-jetson-jetpack6"


def test_environment_reads_git_when_no_override(monkeypatch):
    for var in ("BENCH_GIT_COMMIT", "BENCH_GIT_DIRTY", "BENCH_CONTAINER_IMAGE"):
        monkeypatch.delenv(var, raising=False)
    env = eb.environment()
    assert env["git_commit"] and len(env["git_commit"]) == 40
    assert env["container_image"] is None


# --------------------------------------------------------------------------- failures and explicit INT8


def test_records_default_to_ok_and_failures_show_in_the_table():
    ok = eb.make_record(environment={"host_tag": "jetson"}, command=[], model={}, protocol={},
                        latency_ms=None, accuracy=None, memory=None)
    assert (ok["status"], ok["error"]) == ("ok", None)

    failed = fake_record(host="jetson", precision="int8", gpu="Orin")
    failed.update(status="build_failed", latency_ms=None, accuracy=None)
    failed["environment"]["tensorrt"] = "10.3.0"
    table = eb.markdown_table([fake_record(host="jetson", gpu="Orin"), failed], classes=("Stairs",))
    row = [r for r in table.splitlines() if "| INT8 |" in r][0]
    assert "build failed (TensorRT 10.3.0)" in row
    assert row.count("|") == table.splitlines()[0].count("|")


def test_int8_qdq_is_a_known_precision_with_its_own_engine_name(tmp_path):
    assert eb.PRECISIONS == ("fp32", "fp16", "int8", "int8_qdq")
    assert eb.engine_path(Path("m/yolo26n_nav.pt"), "int8_qdq", "jetson", tmp_path).name == "yolo26n_nav_int8_qdq_jetson.engine"
