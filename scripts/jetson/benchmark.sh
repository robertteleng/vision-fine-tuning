#!/usr/bin/env bash
# Run scripts/benchmark.py on a Jetson inside the pinned Ultralytics JetPack 6 image.
# The container lacks git and nvpmodel, so the host passes commit and power mode in.
# The repo is mounted at the same absolute path so dataset.yaml paths stay valid.
#
# Usage (on the Jetson, from the repo root):
#   scripts/jetson/benchmark.sh --weights models/yolo26n_nav.pt --precisions fp32 fp16 int8
set -euo pipefail

IMAGE="ultralytics/ultralytics:8.4.14-jetson-jetpack6"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  echo "Working tree has uncommitted changes: commit first so the records point to real code." >&2
  exit 1
fi

docker run --rm --runtime nvidia --ipc host --hostname "$(hostname)" \
  -v "$REPO":"$REPO" -w "$REPO" \
  -e BENCH_GIT_COMMIT="$(git -C "$REPO" rev-parse HEAD)" \
  -e BENCH_GIT_DIRTY=0 \
  -e BENCH_POWER_MODE="$(nvpmodel -q 2>/dev/null | head -1)" \
  -e BENCH_CONTAINER_IMAGE="$IMAGE" \
  "$IMAGE" \
  python3 scripts/benchmark.py "$@"
