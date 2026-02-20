"""
Integration tests for the Fine-Tuning Studio pipeline.
"""

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_config_yaml_exists(self, project_root):
        config_path = project_root / "config.yaml"
        assert config_path.exists(), "config.yaml not found"

    def test_config_yaml_valid(self, project_root):
        config_path = project_root / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict)
        assert "model" in config or "epochs" in config

    def test_config_yaml_has_key_fields(self, project_root):
        config_path = project_root / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        for key in ("model", "epochs", "batch", "imgsz"):
            assert key in config, f"config.yaml missing '{key}'"

    def test_dataset_yaml_discovery(self, project_root):
        """Test that find_dataset_yamls works."""
        from src.project import find_dataset_yamls

        yamls = find_dataset_yamls()
        assert isinstance(yamls, list)


class TestModelFiles:
    """Tests for model files."""

    def test_models_dir_exists(self, models_dir):
        if not models_dir.exists():
            pytest.skip("models/ directory not found (fresh clone)")

    def test_find_best_model_function(self):
        """Test that find_best_model from src.inference works."""
        from src.inference import find_best_model

        result = find_best_model()
        if result is not None:
            assert result.suffix in (".pt", ".engine", ".onnx")

    def test_find_available_models(self):
        from src.inference import find_available_models

        models = find_available_models()
        assert isinstance(models, list)


class TestHardwareModule:
    """Tests for src/hardware.py."""

    def test_detect_gpu(self):
        from src.hardware import detect_gpu

        gpu = detect_gpu()
        assert hasattr(gpu, "name")
        assert hasattr(gpu, "vram_mb")
        assert hasattr(gpu, "available")
        assert isinstance(gpu.available, bool)

    def test_suggest_training_defaults(self):
        from src.hardware import suggest_training_defaults

        defaults = suggest_training_defaults()
        assert "batch" in defaults
        assert "workers" in defaults
        assert "imgsz" in defaults
        assert "amp" in defaults

    def test_get_hardware_summary(self):
        from src.hardware import get_hardware_summary

        summary = get_hardware_summary()
        assert isinstance(summary, str)
        assert "GPU" in summary


class TestProjectModule:
    """Tests for src/project.py."""

    def test_get_models_for_task_detect(self):
        from src.project import get_models_for_task

        models = get_models_for_task("detect")
        assert len(models) > 0
        assert all(m.endswith(".pt") for m in models)
        assert "yolo26m.pt" in models

    def test_get_models_for_task_segment(self):
        from src.project import get_models_for_task

        models = get_models_for_task("segment")
        assert all("-seg.pt" in m for m in models)

    def test_get_task_for_model(self):
        from src.project import get_task_for_model

        assert get_task_for_model("yolo26m.pt") == "detect"
        assert get_task_for_model("yolo26m-seg.pt") == "segment"
        assert get_task_for_model("yolo26m-cls.pt") == "classify"

    def test_project_config_from_yaml(self, project_root):
        from src.project import ProjectConfig

        config = ProjectConfig.from_yaml(project_root / "config.yaml")
        assert config.imgsz == 640
        assert config.model == "yolo26m.pt"


class TestGroundingDino:
    """Tests for Grounding DINO auto-annotation."""

    def test_box_to_yolo_function(self, project_root):
        sys.path.insert(0, str(project_root / "scripts"))

        try:
            from auto_annotate_grounding_dino import box_to_yolo
        except ImportError:
            pytest.skip("Grounding DINO script not available")

        box = [400, 400, 600, 600]
        result = box_to_yolo(box, 1000, 1000)

        parts = result.split()
        assert len(parts) == 5
        assert parts[0] == "0"
        assert abs(float(parts[1]) - 0.5) < 0.01
        assert abs(float(parts[2]) - 0.5) < 0.01
        assert abs(float(parts[3]) - 0.2) < 0.01
        assert abs(float(parts[4]) - 0.2) < 0.01


class TestNoHardcoding:
    """Verify no VR/pillar-specific hardcoding remains in src/ or scripts/."""

    def test_no_vr_references_in_src(self, project_root):
        forbidden = ["pillar.yaml", "vr_box", "VR Pillar", "RTX 2060"]
        src_dir = project_root / "src"

        for py_file in src_dir.glob("*.py"):
            content = py_file.read_text()
            for term in forbidden:
                assert term not in content, f"Found '{term}' in {py_file.name}"

    def test_no_vr_references_in_scripts(self, project_root):
        forbidden = ["vr_boxes_", "pillar.yaml", "cajas VR", "Cajas VR"]
        scripts_dir = project_root / "scripts"

        for py_file in scripts_dir.glob("*.py"):
            content = py_file.read_text()
            for term in forbidden:
                assert term not in content, f"Found '{term}' in scripts/{py_file.name}"
