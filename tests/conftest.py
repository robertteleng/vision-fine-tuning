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
