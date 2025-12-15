"""
Tests for split_dataset.py - Dataset splitting functionality.
"""

import tempfile
from pathlib import Path

import pytest

from scripts.split_dataset import (
    get_matching_pairs,
    split_pairs,
)


class TestGetMatchingPairs:
    """Tests for finding image-label pairs."""

    def test_get_matching_pairs_empty_dirs(self, tmp_path):
        """Test with empty directories."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        pairs = get_matching_pairs(images_dir, labels_dir)
        assert pairs == []

    def test_get_matching_pairs_finds_matches(self, tmp_path):
        """Test finding matching image-label pairs."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create matching pairs
        (images_dir / "frame_001.jpg").touch()
        (labels_dir / "frame_001.txt").write_text("0 0.5 0.5 0.1 0.1")

        (images_dir / "frame_002.jpg").touch()
        (labels_dir / "frame_002.txt").write_text("0 0.3 0.3 0.2 0.2")

        pairs = get_matching_pairs(images_dir, labels_dir)
        assert len(pairs) == 2

    def test_get_matching_pairs_ignores_orphans(self, tmp_path):
        """Test that images without labels are ignored."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Image with label
        (images_dir / "frame_001.jpg").touch()
        (labels_dir / "frame_001.txt").write_text("0 0.5 0.5 0.1 0.1")

        # Orphan image (no label)
        (images_dir / "frame_002.jpg").touch()

        # Orphan label (no image)
        (labels_dir / "frame_003.txt").write_text("0 0.5 0.5 0.1 0.1")

        pairs = get_matching_pairs(images_dir, labels_dir)
        assert len(pairs) == 1
        assert pairs[0]['image'].stem == "frame_001"


class TestSplitPairs:
    """Tests for splitting pairs into train/val."""

    def test_split_pairs_ratio(self):
        """Test that split respects the ratio."""
        pairs = [{'image': f'img_{i}', 'label': f'lbl_{i}'} for i in range(100)]

        train, val = split_pairs(pairs, train_ratio=0.8, seed=42)

        assert len(train) == 80
        assert len(val) == 20
        assert len(train) + len(val) == len(pairs)

    def test_split_pairs_deterministic(self):
        """Test that same seed produces same split."""
        pairs = [{'image': f'img_{i}', 'label': f'lbl_{i}'} for i in range(50)]

        train1, val1 = split_pairs(pairs, train_ratio=0.8, seed=42)
        train2, val2 = split_pairs(pairs, train_ratio=0.8, seed=42)

        assert train1 == train2
        assert val1 == val2

    def test_split_pairs_different_seeds(self):
        """Test that different seeds produce different splits."""
        pairs = [{'image': f'img_{i}', 'label': f'lbl_{i}'} for i in range(50)]

        train1, _ = split_pairs(pairs, train_ratio=0.8, seed=42)
        train2, _ = split_pairs(pairs, train_ratio=0.8, seed=123)

        assert train1 != train2

    def test_split_pairs_small_dataset(self):
        """Test split with very small dataset."""
        pairs = [{'image': 'img_0', 'label': 'lbl_0'}]

        train, val = split_pairs(pairs, train_ratio=0.8, seed=42)

        # With 1 item and 0.8 ratio, train gets 0, val gets 1
        assert len(train) + len(val) == 1

    def test_split_pairs_empty(self):
        """Test split with empty list."""
        train, val = split_pairs([], train_ratio=0.8, seed=42)

        assert train == []
        assert val == []
