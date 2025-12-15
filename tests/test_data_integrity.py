"""
Tests for data integrity - verify dataset format and structure.
"""

import re
from pathlib import Path

import pytest


class TestDatasetStructure:
    """Tests for dataset directory structure."""

    def test_dataset_exists(self, data_dir):
        """Test that dataset directory exists."""
        dataset_dir = data_dir / "dataset"
        assert dataset_dir.exists(), "data/dataset/ not found"

    def test_train_structure(self, data_dir):
        """Test train directory structure."""
        train_images = data_dir / "dataset" / "train" / "images"
        train_labels = data_dir / "dataset" / "train" / "labels"

        assert train_images.exists(), "train/images/ not found"
        assert train_labels.exists(), "train/labels/ not found"

    def test_val_structure(self, data_dir):
        """Test val directory structure."""
        val_images = data_dir / "dataset" / "val" / "images"
        val_labels = data_dir / "dataset" / "val" / "labels"

        assert val_images.exists(), "val/images/ not found"
        assert val_labels.exists(), "val/labels/ not found"

    def test_train_has_images(self, data_dir):
        """Test that train has images."""
        train_images = data_dir / "dataset" / "train" / "images"
        if not train_images.exists():
            pytest.skip("Train images directory not found")

        images = list(train_images.glob("*.jpg")) + list(train_images.glob("*.png"))
        assert len(images) > 0, "No images in train/images/"

    def test_val_has_images(self, data_dir):
        """Test that val has images."""
        val_images = data_dir / "dataset" / "val" / "images"
        if not val_images.exists():
            pytest.skip("Val images directory not found")

        images = list(val_images.glob("*.jpg")) + list(val_images.glob("*.png"))
        assert len(images) > 0, "No images in val/images/"


class TestLabelFormat:
    """Tests for YOLO label format."""

    def test_labels_have_correct_format(self, data_dir):
        """Test that label files have correct YOLO format."""
        train_labels = data_dir / "dataset" / "train" / "labels"
        if not train_labels.exists():
            pytest.skip("Train labels directory not found")

        label_files = list(train_labels.glob("*.txt"))
        if not label_files:
            pytest.skip("No label files found")

        # Check first 10 label files
        for label_file in label_files[:10]:
            content = label_file.read_text().strip()
            if not content:
                continue  # Empty labels are OK (no objects)

            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                assert len(parts) >= 5, f"Invalid format in {label_file.name}: {line}"

                # Class ID should be integer
                try:
                    class_id = int(parts[0])
                    assert class_id >= 0, f"Negative class ID in {label_file.name}"
                except ValueError:
                    pytest.fail(f"Invalid class ID in {label_file.name}: {parts[0]}")

                # Coordinates should be floats between 0 and 1
                for i, coord_name in enumerate(['x_center', 'y_center', 'width', 'height'], 1):
                    try:
                        val = float(parts[i])
                        assert 0 <= val <= 1, f"{coord_name} out of range in {label_file.name}: {val}"
                    except ValueError:
                        pytest.fail(f"Invalid {coord_name} in {label_file.name}: {parts[i]}")

    def test_image_label_pairs_match(self, data_dir):
        """Test that each image has corresponding label."""
        train_images = data_dir / "dataset" / "train" / "images"
        train_labels = data_dir / "dataset" / "train" / "labels"

        if not train_images.exists() or not train_labels.exists():
            pytest.skip("Train directories not found")

        images = list(train_images.glob("*.jpg")) + list(train_images.glob("*.png"))
        missing_labels = []

        for img in images[:50]:  # Check first 50
            label_path = train_labels / f"{img.stem}.txt"
            if not label_path.exists():
                missing_labels.append(img.name)

        if missing_labels:
            pytest.fail(f"Images missing labels: {missing_labels[:5]}...")


class TestCoordinateRanges:
    """Tests for coordinate value ranges."""

    def test_coordinates_normalized(self, data_dir):
        """Test that all coordinates are properly normalized (0-1)."""
        train_labels = data_dir / "dataset" / "train" / "labels"
        if not train_labels.exists():
            pytest.skip("Train labels directory not found")

        label_files = list(train_labels.glob("*.txt"))
        out_of_range = []

        for label_file in label_files[:20]:
            content = label_file.read_text().strip()
            if not content:
                continue

            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    for i in range(1, 5):
                        val = float(parts[i])
                        if val < 0 or val > 1:
                            out_of_range.append(f"{label_file.name}:{line_num}")

        assert len(out_of_range) == 0, f"Coordinates out of range: {out_of_range[:5]}"

    def test_box_dimensions_reasonable(self, data_dir):
        """Test that box dimensions are reasonable (not too small/large)."""
        train_labels = data_dir / "dataset" / "train" / "labels"
        if not train_labels.exists():
            pytest.skip("Train labels directory not found")

        label_files = list(train_labels.glob("*.txt"))
        suspicious = []

        for label_file in label_files[:20]:
            content = label_file.read_text().strip()
            if not content:
                continue

            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    width = float(parts[3])
                    height = float(parts[4])

                    # Very small boxes (< 1% of image) might be errors
                    if width < 0.01 or height < 0.01:
                        suspicious.append(f"{label_file.name}: tiny box ({width:.3f}x{height:.3f})")

                    # Very large boxes (> 90% of image) might be errors
                    if width > 0.9 or height > 0.9:
                        suspicious.append(f"{label_file.name}: huge box ({width:.3f}x{height:.3f})")

        # Just warn, don't fail
        if suspicious:
            print(f"\nWarning: Suspicious box dimensions found: {suspicious[:5]}")
