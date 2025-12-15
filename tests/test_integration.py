"""
Integration tests for the training and inference pipeline.
"""

import sys
from pathlib import Path

import pytest
import yaml

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_config_yaml_exists(self, project_root):
        """Test that config.yaml exists."""
        config_path = project_root / "config.yaml"
        assert config_path.exists(), "config.yaml not found"

    def test_config_yaml_valid(self, project_root):
        """Test that config.yaml is valid YAML."""
        config_path = project_root / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict)
        assert 'model' in config or 'epochs' in config

    def test_pillar_yaml_exists(self, data_dir):
        """Test that pillar.yaml exists."""
        yaml_path = data_dir / "pillar.yaml"
        assert yaml_path.exists(), "data/pillar.yaml not found"

    def test_pillar_yaml_valid(self, data_dir):
        """Test that pillar.yaml is valid and has required fields."""
        yaml_path = data_dir / "pillar.yaml"
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert 'path' in config, "pillar.yaml missing 'path'"
        assert 'train' in config, "pillar.yaml missing 'train'"
        assert 'val' in config, "pillar.yaml missing 'val'"
        assert 'names' in config, "pillar.yaml missing 'names'"

    def test_pillar_yaml_dataset_exists(self, data_dir, project_root):
        """Test that dataset paths in pillar.yaml exist."""
        yaml_path = data_dir / "pillar.yaml"
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Try both the configured path and relative to project
        dataset_path = Path(config['path'])
        if not dataset_path.exists():
            # Try relative path within project
            dataset_path = data_dir / "dataset"

        if not dataset_path.exists():
            pytest.skip(f"Dataset not found at {config['path']} or {data_dir / 'dataset'}")

        # Check train images exist
        train_images = dataset_path / config['train']
        assert train_images.exists(), f"Train path not found: {train_images}"

        # Check val images exist
        val_images = dataset_path / config['val']
        assert val_images.exists(), f"Val path not found: {val_images}"


class TestModelFiles:
    """Tests for model files."""

    def test_models_dir_exists(self, models_dir):
        """Test that models directory exists."""
        assert models_dir.exists(), "models/ directory not found"

    def test_has_trained_model(self, models_dir):
        """Test that at least one trained model exists."""
        if not models_dir.exists():
            pytest.skip("models/ directory not found")

        models = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.engine"))
        assert len(models) > 0, "No trained models found in models/"

    def test_model_naming_convention(self, models_dir):
        """Test that models follow naming convention."""
        if not models_dir.exists():
            pytest.skip("models/ directory not found")

        pt_models = list(models_dir.glob("*_pillars_*.pt"))
        # Should have at least one model with correct naming
        # Format: {base}_pillars_{date}.pt or {base}+pillars_{date}.pt
        if pt_models:
            for model in pt_models:
                name = model.stem
                assert "pillars" in name, f"Model {model.name} doesn't follow naming convention"


class TestInferenceScript:
    """Tests for inference.py script."""

    def test_find_best_model_function(self, project_root, models_dir):
        """Test that find_best_model works."""
        if not models_dir.exists() or not list(models_dir.glob("*.pt")):
            pytest.skip("No models available")

        sys.path.insert(0, str(project_root / "scripts"))
        from inference import find_best_model

        model_path = find_best_model()
        assert model_path.exists()
        assert model_path.suffix in ['.pt', '.engine', '.onnx']


class TestGroundingDino:
    """Tests for Grounding DINO auto-annotation."""

    def test_box_to_yolo_function(self, project_root):
        """Test box_to_yolo conversion."""
        sys.path.insert(0, str(project_root / "scripts"))

        try:
            from auto_annotate_grounding_dino import box_to_yolo
        except ImportError:
            pytest.skip("Grounding DINO script not available")

        # Test box at center of 1000x1000 image
        box = [400, 400, 600, 600]  # x1, y1, x2, y2
        result = box_to_yolo(box, 1000, 1000)

        parts = result.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # class

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        assert abs(x_center - 0.5) < 0.01
        assert abs(y_center - 0.5) < 0.01
        assert abs(width - 0.2) < 0.01
        assert abs(height - 0.2) < 0.01
