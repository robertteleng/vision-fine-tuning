"""Project configuration and YOLO model registry."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Model registry — all supported YOLO families, sizes, and tasks
# ---------------------------------------------------------------------------

YOLO_FAMILIES = ["yolo26", "yolo12", "yolov11", "yolov8"]

YOLO_SIZES = ["n", "s", "m", "l", "x"]

YOLO_TASKS = {
    "detect": "",
    "segment": "-seg",
    "classify": "-cls",
    "pose": "-pose",
    "obb": "-obb",
}


def get_models_for_task(task: str = "detect", families: list[str] | None = None) -> list[str]:
    """List available pretrained model names for a task.

    >>> get_models_for_task("detect", ["yolo26"])
    ['yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt', 'yolo26x.pt']
    """
    suffix = YOLO_TASKS.get(task, "")
    families = families or YOLO_FAMILIES
    return [f"{fam}{size}{suffix}.pt" for fam in families for size in YOLO_SIZES]


def get_task_for_model(model_name: str) -> str:
    """Infer the task from a model filename.

    >>> get_task_for_model("yolo26m-seg.pt")
    'segment'
    """
    stem = Path(model_name).stem  # e.g. "yolo26m-seg"
    for task, suffix in YOLO_TASKS.items():
        if suffix and suffix in stem:
            return task
    return "detect"


# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Runtime configuration for a training/inference session."""

    # Dataset
    data_yaml: str = ""
    nc: int = 1
    names: dict[int, str] = field(default_factory=lambda: {0: "object"})

    # Model
    model: str = "yolo26m.pt"
    task: str = "detect"

    # Training
    epochs: int = 100
    batch: int = -1
    imgsz: int = 640
    patience: int = 20

    # Paths
    project_dir: str = "runs/train"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        """Load config from a YAML file, ignoring unknown keys."""
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)


def find_dataset_yamls(search_dir: Path | None = None) -> list[Path]:
    """Find all dataset YAML files under a directory."""
    search_dir = search_dir or (PROJECT_ROOT / "data")
    if not search_dir.exists():
        return []
    return sorted(search_dir.rglob("*.yaml"))
