"""Pytest fixtures for ReelMotion AI test suite."""

import os
import sys
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import get_settings
from backend.main import create_app


@pytest.fixture
def app():
    """Create a FastAPI test app instance."""
    return create_app()


@pytest.fixture
def client(app) -> Generator[TestClient, None, None]:
    """TestClient fixture with app lifespan handled."""
    with TestClient(app) as test_client:
        yield test_client
