# Experiment: can zero-shot auto-annotation fix `Stairs`?

**Status: pre-registered on 2026-09-16, before any of these models was trained or any
pseudo-label was generated.** Results are appended below the line at the end, the design above it is
not edited after the fact. Changes to the plan, if any, get dated notes.

## Why

`Stairs` is the class that matters most for a blind pedestrian and one of the weakest (mAP50 0.318
for YOLO26s). The training set has 484 images with stairs. Hand-labelling more is slow, and
Grounding DINO can label from a text prompt. The question is what that shortcut costs.

## Question

For the edge model (YOLO26 **nano**), how much of the `Stairs` gain from adding more stairs images
does **zero-shot auto-annotation** capture, compared with **human** labels on the same images?

## Design

Three models, identical recipe (`config.yaml`: 50 epochs, seed 42, COCO-pretrained `yolo26n.pt`),
trained from scratch each time. Only the extra data differs.

| Arm | Training data |
|---|---|
| **A** baseline | `nav_combined` (already trained) |
| **B** auto | `nav_combined` + *K* extra images, `Stairs` boxes from **Grounding DINO** |
| **C** human | `nav_combined` + the same *K* images, `Stairs` boxes from **Open Images annotators** |

- **Extra images (*K* = 1000):** Open Images V7 **train** images that have at least one non-group
  `Stairs` box and are not in `data_manifest/`. Take the first 1,000 by image ID and pin them in
  `data_manifest/experiment_stairs_train.txt`. The pool holds 4,049 unused stairs images.
- **Both arms label only `Stairs`** on the extra images. Other objects in them stay unlabelled in
  both arms, so the label source is the only difference.
- **Grounding DINO settings, fixed now:** `IDEA-Research/grounding-dino-tiny`, prompt `"stairs."`,
  score threshold 0.30 (the default of `scripts/auto_annotate_grounding_dino.py`). No prompt or
  threshold search: tuning them against human labels would turn arm B into a semi-supervised arm.

## Evaluation

- **Stairs test set (primary):** the validation split has only 45 `Stairs` instances, too few to
  separate the arms. Evaluation uses Open Images V7 **test** split images with `Stairs` boxes, up
  to 500 images (first by ID), never used for training or calibration, pinned in
  `data_manifest/experiment_stairs_test.txt`. Only `Stairs` AP is read on it.
- **Validation split (secondary):** global mAP50 and every per-class mAP50, to catch regressions.
- **Pseudo-label quality:** precision and recall of the Grounding DINO boxes against the human
  boxes on the *K* images at IoU ≥ 0.5.
- Settings: `imgsz=640`, `batch=1`, `conf=0.001`, `iou=0.7` (as in `docs/BENCHMARK_METHODOLOGY.md`).

## Decision criterion (fixed now)

Let `gain(X) = Stairs AP50 on the test set of arm X − arm A`.

1. **Auto-annotation is worth using for `Stairs`** if `gain(B) ≥ 0.5 × gain(C)`, `gain(C) ≥ 0.02`
   and arm B's global validation mAP50 is at most 0.01 below arm A.
2. If `gain(C) < 0.02`, more data of this kind does not help `Stairs` at all. The bottleneck is
   elsewhere (domain, box ambiguity, model size), and no verdict on auto-annotation is drawn.
3. If `gain(B) < 0`, the pseudo-labels hurt.

## Threats to validity

- **Domain:** Open Images photos are not egocentric walking footage like AriaGuard's camera. A win
  here is a win on photos.
- **Partial labels:** the extra images contain unlabelled people, cars, doors… which training
  treats as background. It affects B and C equally, but it can lower other classes; the
  validation check exists for that.
- **One seed per arm:** differences of ~0.01 AP are within run-to-run noise and are reported as such.
- **Group boxes:** Open Images marks some boxes as `IsGroupOf` (several steps boxed together).
  Grounding DINO draws its own boxes, which may split or merge differently from the annotators.
- **Nano only.** The small model is not retrained.

## Notes during execution

- **2026-09-16, test set size.** The Open Images V7 test split has only 131 images with a `Stairs`
  box, 120 of them with at least one non-group box. The test set is therefore those 120 images,
  with 140 instances; the design said "up to 500". It is still about 3× the 45 validation
  instances, but differences of a few hundredths remain noisy. The *K* = 1,000 extra training
  images carry 1,291 human `Stairs` boxes (77 `IsGroupOf`).

---

## Results

**Run on 2026-09-17** (commit of the tooling `6642e46`; numbers from `benchmarks/experiment_stairs/`).

### Pseudo-label quality

Grounding DINO drew **1,554** boxes on the 1,000 extra images against **1,291** human boxes. At IoU ≥ 0.5:
**precision 0.594, recall 0.715**. 17 images got no box at all.

### Models

| Arm | Stairs AP50 (test, 140 inst.) | Stairs AP50-95 (test) | Global mAP50 (val) | Stairs mAP50 (val, 45 inst.) |
|---|---|---|---|---|
| A · baseline | 0.464 | 0.240 | 0.396 | 0.316 |
| B · + auto labels | 0.543 | 0.326 | 0.398 | 0.443 |
| C · + human labels | **0.590** | **0.331** | **0.411** | 0.381 |

`gain(B) = +0.079`, `gain(C) = +0.126`. Arm B's global validation mAP50 is **0.001 above** A.

### Verdict (pre-registered criterion)

`gain(C) = 0.126 ≥ 0.02` and `gain(B) = 0.079 ≥ 0.5 × gain(C) = 0.063`, with no validation regression →
**auto-annotation is worth using for `Stairs`**. Zero-shot labels captured **63 %** of the gain that
human labels on the same images deliver.

### Reading the result

- **The boxes themselves are nearly as good as human ones.** At the stricter AP50-95, B and C are within 0.005.
  Most of what B loses comes from spurious and missed stairs (precision 0.59, recall 0.72 against
  humans), not from sloppy box geometry.
- **More stairs data helps a lot**: +0.126 AP50 with human labels, from 484 to 1,484 training
  images with stairs. The class was data-starved.
- Human labels also lifted global mAP50 by 0.015. Auto labels were neutral.
- **Limits, as pre-registered:** one seed per arm; 140 test instances; Open Images photos, not egocentric
  footage. On validation, B scores above C on `Stairs` (0.443 vs 0.381), which shows how noisy the 45
  instances are and why the test set was the primary metric.
