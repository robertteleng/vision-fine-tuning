"""Project paths and the base models this pipeline supports."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# YOLO26 is NMS-free end to end, which keeps post-processing out of the edge
# latency budget. Nano is the Jetson target; small is the accuracy reference.
BASE_MODELS = ("yolo26n.pt", "yolo26s.pt")


def find_dataset_yamls(search_dir: Path | None = None) -> list[Path]:
    """Find all dataset YAML files under a directory."""
    search_dir = search_dir or (PROJECT_ROOT / "data")
    if not search_dir.exists():
        return []
    return sorted(search_dir.rglob("*.yaml"))
