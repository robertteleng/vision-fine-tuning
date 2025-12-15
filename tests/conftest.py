"""
Pytest configuration and fixtures.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    """Return project root path."""
    return PROJECT_ROOT


@pytest.fixture
def data_dir(project_root):
    """Return data directory path."""
    return project_root / "data"


@pytest.fixture
def models_dir(project_root):
    """Return models directory path."""
    return project_root / "models"


@pytest.fixture
def sample_image(data_dir):
    """Return path to a sample image for testing."""
    # Try to find a sample image in the dataset
    dataset_dir = data_dir / "dataset" / "train" / "images"
    if dataset_dir.exists():
        images = list(dataset_dir.glob("*.jpg"))
        if images:
            return images[0]
    return None


@pytest.fixture
def sample_label(data_dir):
    """Return path to a sample label for testing."""
    dataset_dir = data_dir / "dataset" / "train" / "labels"
    if dataset_dir.exists():
        labels = list(dataset_dir.glob("*.txt"))
        if labels:
            return labels[0]
    return None
