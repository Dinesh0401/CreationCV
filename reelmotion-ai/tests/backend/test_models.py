"""Tests for model manifest management and model API endpoints."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from backend.services.model_manager import ModelManager


def test_model_manager_loads_manifest():
    """Test loading of model_manifest.yaml."""
    mgr = ModelManager()
    models = mgr.list_models()
    assert len(models) > 0
    # Check that BiRefNet and Wan Animate are present
    model_ids = [m.id for m in models]
    assert "birefnet-general" in model_ids


def test_models_endpoint(client: TestClient):
    """Test GET /api/models."""
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert data["total"] > 0


def test_model_not_found(client: TestClient):
    """Failure test: Non-existent model ID returns 404."""
    res = client.get("/api/models/non-existent-model-id-999")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_missing_manifest_fallback(tmp_path: Path):
    """Failure test: ModelManager with non-existent manifest path initializes empty list."""
    missing_manifest = tmp_path / "missing_manifest.yaml"
    mgr = ModelManager(manifest_path=missing_manifest)
    assert mgr.list_models() == []

