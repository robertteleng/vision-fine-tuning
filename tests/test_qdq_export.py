"""Tests for the GPU-free parts of src/qdq_export.py."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.qdq_export import letterbox_tensor, qdq_onnx_path  # noqa: E402


def test_qdq_onnx_path():
    assert qdq_onnx_path(Path("models/yolo26s_nav.pt"), Path("models/qdq")) == Path("models/qdq/yolo26s_nav_int8_qdq.onnx")


def test_letterbox_tensor_matches_engine_input():
    image = np.zeros((480, 960, 3), dtype=np.uint8)
    image[..., 2] = 255  # pure red in BGR
    t = letterbox_tensor(image, 640)
    assert t.shape == (3, 640, 640) and t.dtype == np.float32
    assert 0.0 <= t.min() and t.max() <= 1.0
    assert t[0, 320, 320] == 1.0 and t[2, 320, 320] == 0.0  # channel 0 is red after BGR -> RGB
    assert np.isclose(t[1, 5, 320], 114 / 255)  # letterbox padding colour (image is wider than tall)
