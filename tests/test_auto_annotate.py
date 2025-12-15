"""
Tests for auto_annotate.py - Template matching auto-annotation.
"""

import numpy as np
import pytest

from scripts.auto_annotate import (
    load_template,
    match_template_multiscale,
    nms,
    to_yolo_format,
)


class TestNMS:
    """Tests for Non-Maximum Suppression."""

    def test_nms_empty_list(self):
        """NMS should handle empty detection list."""
        result = nms([])
        assert result == []

    def test_nms_single_detection(self):
        """NMS should keep single detection."""
        detections = [{'x': 10, 'y': 10, 'w': 50, 'h': 50, 'conf': 0.9}]
        result = nms(detections)
        assert len(result) == 1

    def test_nms_no_overlap(self):
        """NMS should keep non-overlapping detections."""
        detections = [
            {'x': 0, 'y': 0, 'w': 50, 'h': 50, 'conf': 0.9},
            {'x': 100, 'y': 100, 'w': 50, 'h': 50, 'conf': 0.8},
        ]
        result = nms(detections, iou_threshold=0.3)
        assert len(result) == 2

    def test_nms_removes_duplicates(self):
        """NMS should remove highly overlapping detections."""
        detections = [
            {'x': 10, 'y': 10, 'w': 50, 'h': 50, 'conf': 0.9},
            {'x': 12, 'y': 12, 'w': 50, 'h': 50, 'conf': 0.7},  # Almost same box
        ]
        result = nms(detections, iou_threshold=0.3)
        assert len(result) == 1
        assert result[0]['conf'] == 0.9  # Keep higher confidence

    def test_nms_keeps_highest_confidence(self):
        """NMS should always keep the highest confidence detection."""
        detections = [
            {'x': 10, 'y': 10, 'w': 50, 'h': 50, 'conf': 0.5},
            {'x': 10, 'y': 10, 'w': 50, 'h': 50, 'conf': 0.9},
            {'x': 10, 'y': 10, 'w': 50, 'h': 50, 'conf': 0.7},
        ]
        result = nms(detections, iou_threshold=0.3)
        assert len(result) == 1
        assert result[0]['conf'] == 0.9


class TestYOLOFormat:
    """Tests for YOLO format conversion."""

    def test_to_yolo_format_center(self):
        """Test conversion to YOLO normalized format."""
        detections = [{'x': 100, 'y': 100, 'w': 200, 'h': 100}]
        result = to_yolo_format(detections, img_width=1000, img_height=1000)

        assert len(result) == 1
        parts = result[0].split()
        assert len(parts) == 5

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        assert class_id == 0
        assert 0 <= x_center <= 1
        assert 0 <= y_center <= 1
        assert 0 <= width <= 1
        assert 0 <= height <= 1

    def test_to_yolo_format_normalization(self):
        """Test that coordinates are properly normalized."""
        # Detection at center of 640x480 image
        detections = [{'x': 270, 'y': 190, 'w': 100, 'h': 100}]
        result = to_yolo_format(detections, img_width=640, img_height=480)

        parts = result[0].split()
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # x_center should be (270 + 50) / 640 = 0.5
        assert abs(x_center - 0.5) < 0.01
        # y_center should be (190 + 50) / 480 = 0.5
        assert abs(y_center - 0.5) < 0.01
        # width should be 100 / 640 = 0.15625
        assert abs(width - 0.15625) < 0.001
        # height should be 100 / 480 = 0.208333
        assert abs(height - 0.208333) < 0.001

    def test_to_yolo_format_empty(self):
        """Test with no detections."""
        result = to_yolo_format([], img_width=640, img_height=480)
        assert result == []


class TestMatchTemplate:
    """Tests for template matching."""

    def test_match_template_with_pattern(self):
        """Test matching with a patterned template (more realistic)."""
        # Create a template with a pattern (checkerboard-like)
        template = np.zeros((50, 50), dtype=np.uint8)
        template[0:25, 0:25] = 255
        template[25:50, 25:50] = 255

        # Create image with template embedded
        image = np.zeros((200, 200), dtype=np.uint8)
        image[75:125, 75:125] = template

        detections = match_template_multiscale(
            image, template,
            scales=[1.0],
            threshold=0.9
        )

        # Should find at least one detection
        assert len(detections) >= 1

        # Best detection should be near where we placed it
        if detections:
            # Sort by confidence
            best = max(detections, key=lambda d: d['conf'])
            assert 70 <= best['x'] <= 80
            assert 70 <= best['y'] <= 80

    def test_match_template_returns_detections_structure(self):
        """Test that detections have correct structure."""
        template = np.random.randint(0, 255, (30, 30), dtype=np.uint8)
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        detections = match_template_multiscale(
            image, template,
            scales=[1.0],
            threshold=0.5
        )

        # Each detection should have required keys
        for det in detections:
            assert 'x' in det
            assert 'y' in det
            assert 'w' in det
            assert 'h' in det
            assert 'conf' in det
            assert 0 <= det['conf'] <= 1
