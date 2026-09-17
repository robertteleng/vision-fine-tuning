# Benchmark methodology

How every speed and accuracy number in this repo is produced, so that numbers from
different machines can be compared and anyone can reproduce them.

The protocol lives in code (`src/edge_bench.py`, `src/trt_export.py`,
`scripts/benchmark.py`). This page explains the choices behind it.

## Question

For a 24-class obstacle detector that has to run on a wearable-class device:

1. How fast is it end to end, including the tail (p95/p99), not just on average?
2. How much accuracy does each precision cost, **per class**? A drop hidden in
   `Stairs` matters more than the global mAP.
3. Does INT8 earn its place over FP16 on each device?

## What is compared

| Axis | Values |
|---|---|
| Model | YOLO26 **nano** (edge target) and **small** (accuracy reference), same data and recipe |
| Precision | **FP32** (PyTorch weights), **FP16** and **INT8** (TensorRT engines) |
| Device | x86 + RTX 5060 Ti; Jetson Orin Nano Super (MAXN_SUPER) |

## Protocol

### Accuracy

- Full validation split of `data/nav_combined` (2,571 images; `scripts/build_dataset.py` rebuilds it identically).
- `imgsz=640`, `batch=1`, `conf=0.001`, `iou=0.7`. The IoU matches what training
  validated with, so benchmark mAP is comparable with the training curves.
- Reported: precision, recall, mAP50, mAP50-95, and mAP50 / mAP50-95 **per class**.
- Every engine is scored itself. Accuracy is never copied from the PyTorch model.
- Batch 1 on purpose: the engines have a static batch-1 shape, and scoring everything the same
  way keeps the precisions comparable. Numbers differ slightly from the validation printed during
  training, which runs batched with rectangular padding (e.g. nano `Stairs` mAP50 0.316 here vs
  0.330 in the training log). Compare only numbers produced by the same protocol.

### Latency

- 500 validation images, chosen with a fixed seed from the name-sorted list, so every machine
  times the same images. They are decoded and held in memory before timing: disk I/O and JPEG
  decoding are not part of the measurement.
- 50 warm-up inferences first (CUDA context, cuDNN/TensorRT autotuning, allocator growth).
- Batch 1, `imgsz=640`, `conf=0.25` (a deployment-like threshold: post-processing cost depends
  on how many boxes survive).
- Each image is timed with `time.perf_counter()` around `model.predict(...)` followed by
  `torch.cuda.synchronize()`, so asynchronous GPU work is not left out of the interval.
- Reported for **end to end** and for Ultralytics' pre-process / inference / post-process split:
  mean, std, min, p50, p90, p95, p99, max, and throughput = 1000 / mean.
- **Engine only** (TensorRT artifacts): the deserialized engine executed through the TensorRT
  Python API with its input tensor already on the GPU, the same number of timed iterations and
  warm-up, `torch.cuda.synchronize()` after each run. No pre/post-processing, no host copies, no
  Ultralytics code. The gap between this and end to end is what the surrounding pipeline costs.
  On the Jetson it is most of the time (added 2026-09-17, after `trtexec` showed nano and small
  engines at 4.2 and 6.8 ms while the Python pipeline reported 23.6 ms for both).
- The benchmark refuses to run while another process uses the GPU (for example a training run).
  `--allow-busy-gpu` exists but the resulting numbers are not published.

### Artifacts

- **TensorRT engines are built on the device that runs them.** They are not portable between GPU
  architectures or TensorRT versions, so each machine exports its own and the file name carries
  the host (`models/engines/<model>_<precision>_<host>.engine`).
- Static input shape (1×3×640×640), ONNX simplified before building.
- The weights' SHA-256, engine size and build time go into the record.

### INT8 calibration

- **Calibration uses 1,000 TRAIN images** (fixed seed). By default Ultralytics 8.4 calibrates on
  the `val` split of the dataset it receives, the same images the engine is later scored on.
  That would leak the evaluation set into the quantization ranges.
- Calibrator: TensorRT **MinMax**, the Ultralytics default (Entropy is only used with DLA).
  MinMax keeps the full observed range, so a few extreme activations can cost resolution for
  everything else.
- Ultralytics sets only the INT8 builder flag. **Layers TensorRT cannot quantize fall back to
  FP32, not FP16**, which can make INT8 barely faster than FP16. This is measured as is, not
  patched.
- Ultralytics reuses a `.cache` calibration file if it finds one. Engines are built in an empty
  scratch directory so a stale cache from another calibration set can never be picked up.

### INT8 decision criterion

Written on **2026-09-16, before any INT8 number existed** (`INT8_CRITERION` in
`src/edge_bench.py`). On a given device, INT8 replaces FP16 only if all three hold:

1. p95 end-to-end latency at least **20 %** lower than FP16;
2. global mAP50 at most **0.01** lower than FP16;
3. `Stairs` mAP50 at most **0.02** lower than FP16.

Fixing the bar before measuring is what keeps the decision from being fitted to the result.

## Records

Each (model, precision, device) run writes one JSON file to `benchmarks/results/` with:

- **environment**: host, GPU, NVIDIA driver, CUDA, cuDNN, TensorRT, PyTorch, Ultralytics, ONNX,
  Python, CPU; on Jetson also the L4T release and power mode; git commit and whether the tree was dirty;
- **command** as run;
- **model**: weights path and SHA-256, precision, artifact path and size, calibration details, build time;
- **protocol** settings;
- **latency_ms**, **accuracy** and **memory** (GPU memory on a discrete GPU, system RAM on
  Jetson, which has unified memory; before/after loading, approximate).

`scripts/make_tables.py` builds the README tables and the INT8 decisions **from these files
only**. Tables are never edited by hand. When a configuration is re-run, the newest record wins.

## Known limits

- **Small classes are noisy.** In validation, `Stairs` has 45 instances in 36 images and
  `Street light` 40 instances in only 7 images. Differences of a few hundredths in their mAP
  are within noise and should not drive a decision alone.
- **One run per configuration.** Latency percentiles come from 500 timed inferences, but
  run-to-run drift (thermal state, background load) is not averaged across runs.
- **End-to-end means Ultralytics' Python pipeline**, not a C++ TensorRT runtime. The inference
  column isolates the engine. A production C++ path would have lower pre- and post-processing cost.
- **Memory figures are approximate** (whole-device readings, not per-process peaks).
- **Jetson frequency scaling stays on.** The device runs in `MAXN_SUPER` (recorded) without
  `jetson_clocks`, so CPU/GPU frequencies can still scale down between bursts. That is the
  realistic deployment setting, but it can widen the latency tail.
- **TensorRT bugs shape what can be measured, and failures are recorded.** A build that fails is
  written as a `build_failed` record with TensorRT's own error:
  - Ultralytics' implicit INT8 does not build on **TensorRT 10.3** (Jetson): internal
    `checkSanity::checkLinks` assertion during calibration.
  - The explicit Q/DQ INT8 ONNX does not build on **TensorRT 10.16** with the RTX 5060 Ti (Blackwell,
    sm_120): `MyelinCheckException` in Myelin's code generator, a known issue on Blackwell
    ([NVIDIA/TensorRT#4743](https://github.com/NVIDIA/TensorRT/issues/4743)).
  - The same Q/DQ ONNX builds and runs on the Jetson. Neither bug was worked around; the published
    decision (FP16) would not change.
- **Software versions.** Everything the project controls is identical on every device: dataset,
  weights, benchmark code (git commit) and **Ultralytics 8.4.14** (`uv.lock` on x86, the pinned
  `ultralytics/ultralytics:8.4.14-jetson-jetpack6` image on Jetson). TensorRT, CUDA and the driver
  are set by the platform and are recorded, not matched: the RTX 5060 Ti (Blackwell) needs
  TensorRT ≥ 10.8, while JetPack 6.2 ships TensorRT 10.3.

## Reproduce

```bash
# x86 + NVIDIA GPU
uv sync --extra datasets --extra export
uv run python scripts/build_dataset.py --out data/nav_combined --download
uv run python scripts/benchmark.py --weights models/yolo26n_nav.pt models/yolo26s_nav.pt --precisions fp32 fp16 int8
uv run python scripts/make_tables.py
```

```bash
# Jetson (JetPack 6.2), inside the pinned Ultralytics image; commit first, the script refuses a dirty tree
scripts/jetson/benchmark.sh --weights models/yolo26n_nav.pt models/yolo26s_nav.pt --precisions fp32 fp16 int8
```
