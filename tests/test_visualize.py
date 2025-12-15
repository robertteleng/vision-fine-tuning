"""
Tests for visualize_annotations.py - Annotation visualization.
"""

import tempfile
from pathlib import Path

import pytest

from scripts.visualize_annotations import (
    load_annotations,
    yolo_to_pixels,
)


class TestLoadAnnotations:
    """Tests for loading YOLO annotations."""

    def test_load_annotations_valid(self, tmp_path):
        """Test loading valid annotation file."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.1 0.2\n1 0.3 0.7 0.15 0.25")

        annotations = load_annotations(label_file)

        assert len(annotations) == 2
        assert annotations[0]['class'] == 0
        assert annotations[0]['x'] == 0.5
        assert annotations[0]['y'] == 0.5
        assert annotations[0]['w'] == 0.1
        assert annotations[0]['h'] == 0.2

        assert annotations[1]['class'] == 1
        assert annotations[1]['x'] == 0.3

    def test_load_annotations_empty_file(self, tmp_path):
        """Test loading empty annotation file."""
        label_file = tmp_path / "empty.txt"
        label_file.write_text("")

        annotations = load_annotations(label_file)
        assert annotations == []

    def test_load_annotations_nonexistent(self, tmp_path):
        """Test loading non-existent file."""
        annotations = load_annotations(tmp_path / "nonexistent.txt")
        assert annotations == []

    def test_load_annotations_with_blank_lines(self, tmp_path):
        """Test loading file with blank lines."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.1 0.2\n\n\n1 0.3 0.7 0.15 0.25\n")

        annotations = load_annotations(label_file)
        assert len(annotations) == 2

    def test_load_annotations_with_extra_whitespace(self, tmp_path):
        """Test loading file with extra whitespace."""
        label_file = tmp_path / "test.txt"
        label_file.write_text("  0 0.5 0.5 0.1 0.2  \n")

        annotations = load_annotations(label_file)
        assert len(annotations) == 1


class TestYoloToPixels:
    """Tests for converting YOLO coords to pixels."""

    def test_yolo_to_pixels_center(self):
        """Test conversion of centered box."""
        ann = {'x': 0.5, 'y': 0.5, 'w': 0.2, 'h': 0.2}
        x1, y1, x2, y2 = yolo_to_pixels(ann, img_width=1000, img_height=1000)

        # Center at 500,500 with size 200x200
        assert x1 == 400
        assert y1 == 400
        assert x2 == 600
        assert y2 == 600

    def test_yolo_to_pixels_corner(self):
        """Test conversion of corner box."""
        ann = {'x': 0.1, 'y': 0.1, 'w': 0.2, 'h': 0.2}
        x1, y1, x2, y2 = yolo_to_pixels(ann, img_width=1000, img_height=1000)

        # Center at 100,100 with size 200x200
        assert x1 == 0
        assert y1 == 0
        assert x2 == 200
        assert y2 == 200

    def test_yolo_to_pixels_rectangular(self):
        """Test conversion with non-square image."""
        ann = {'x': 0.5, 'y': 0.5, 'w': 0.5, 'h': 0.5}
        x1, y1, x2, y2 = yolo_to_pixels(ann, img_width=640, img_height=480)

        # Center at 320,240 with size 320x240
        assert x1 == 160  # 320 - 160
        assert y1 == 120  # 240 - 120
        assert x2 == 480  # 320 + 160
        assert y2 == 360  # 240 + 120

    def test_yolo_to_pixels_small_box(self):
        """Test conversion of very small box."""
        ann = {'x': 0.5, 'y': 0.5, 'w': 0.01, 'h': 0.01}
        x1, y1, x2, y2 = yolo_to_pixels(ann, img_width=1000, img_height=1000)

        # Size should be 10x10 pixels
        assert x2 - x1 == 10
        assert y2 - y1 == 10

    def test_yolo_to_pixels_full_image(self):
        """Test conversion of box covering entire image."""
        ann = {'x': 0.5, 'y': 0.5, 'w': 1.0, 'h': 1.0}
        x1, y1, x2, y2 = yolo_to_pixels(ann, img_width=640, img_height=480)

        assert x1 == 0
        assert y1 == 0
        assert x2 == 640
        assert y2 == 480
