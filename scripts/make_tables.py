#!/usr/bin/env python3
"""
Build the results tables and INT8 decisions from benchmarks/results/*.json.

Numbers in the README come only from here. --readme rewrites the block
between the BENCHMARK markers in README.md.

Usage:
    uv run python scripts/make_tables.py
    uv run python scripts/make_tables.py --readme
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import edge_bench as eb  # noqa: E402

START, END = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"


def int8_lines(records) -> list[str]:
    latest = eb.latest_by_key(records)
    lines = []
    for (host, name, precision), r in sorted(latest.items()):
        if precision not in ("int8", "int8_qdq") or (host, name, "fp16") not in latest:
            continue
        fp16 = latest[(host, name, "fp16")]
        device = r["environment"].get("gpu") or host
        if r.get("status") == "build_failed":
            lines.append(f"- **{device} · {name} · {precision.upper()}**: engine could not be built")
            continue
        if not (r.get("accuracy") and fp16.get("accuracy")):
            continue
        v = eb.int8_verdict(fp16, r)
        checks = ", ".join(f"{k} {c['value']:+.3f} {'✓' if c['pass'] else '✗'}" for k, c in v["checks"].items())
        lines.append(f"- **{device} · {name} · {precision.upper()}**: {'use it' if v['use_int8'] else 'keep FP16'} ({checks})")
    return lines


def render(records) -> str:
    text = eb.markdown_table(records)
    decisions = int8_lines(records)
    if decisions:
        c = eb.INT8_CRITERION
        text += (f"\nINT8 criterion (fixed on {c['decided_on']}, before measuring): p95 at least "
                 f"{c['min_p95_latency_reduction']:.0%} lower, mAP50 drop at most {c['max_map50_drop']}, "
                 f"Stairs mAP50 drop at most {c['max_stairs_map50_drop']}.\n\n" + "\n".join(decisions) + "\n")
    return text


def main():
    parser = argparse.ArgumentParser(description="Navigation detector — results tables")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "benchmarks/results")
    parser.add_argument("--readme", action="store_true", help="Rewrite the table block in README.md")
    args = parser.parse_args()

    records = eb.load_records(args.results)
    if not records:
        sys.exit(f"no records in {args.results}")
    text = render(records)
    if not args.readme:
        print(text)
        return
    readme = PROJECT_ROOT / "README.md"
    content = readme.read_text()
    if START not in content or END not in content:
        sys.exit(f"README.md has no {START} ... {END} block")
    head, rest = content.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{text}{END}{tail}")
    print(f"README.md updated from {len(records)} records")


if __name__ == "__main__":
    main()
