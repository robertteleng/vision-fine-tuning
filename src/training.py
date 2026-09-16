"""Read training results produced by Ultralytics runs."""

import csv
from pathlib import Path

from src.hardware import detect_gpu, get_hardware_summary
from src.project import PROJECT_ROOT

RUNS_DIR = PROJECT_ROOT / "runs" / "train"


def get_training_metrics():
    """Read metrics from the latest training run dynamically."""
    if not RUNS_DIR.exists():
        return "No training runs found. Train a model first."

    experiments = sorted(RUNS_DIR.glob("*/"), reverse=True)
    if not experiments:
        return "No training runs found. Train a model first."

    # Find latest run with results.csv
    for exp in experiments:
        results_csv = exp / "results.csv"
        if not results_csv.exists():
            continue

        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        last = rows[-1]

        # Ultralytics CSV column names (strip whitespace)
        clean = {k.strip(): v.strip() for k, v in last.items()}

        # Extract metrics (column names vary slightly across versions)
        def get_metric(keys, fmt=".3f"):
            for k in keys:
                if k in clean:
                    try:
                        val = float(clean[k])
                        return f"{val:{fmt}}"
                    except ValueError:
                        pass
            return "N/A"

        precision = get_metric(["metrics/precision(B)", "metrics/precision"])
        recall = get_metric(["metrics/recall(B)", "metrics/recall"])
        map50 = get_metric(["metrics/mAP50(B)", "metrics/mAP50"])
        map50_95 = get_metric(["metrics/mAP50-95(B)", "metrics/mAP50-95"])

        gpu = detect_gpu()
        hw = get_hardware_summary(gpu)

        return f"""## Latest Training: `{exp.name}`

| Metric | Value |
|--------|-------|
| Precision | {precision} |
| Recall | {recall} |
| mAP@50 | {map50} |
| mAP@50-95 | {map50_95} |
| Epochs | {len(rows)} |

{hw}
"""

    return "No results.csv found in any training run."
