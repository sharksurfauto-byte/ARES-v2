"""Unit tests for Phase 6 FastAPI Visualizer Backend Endpoints."""

import pytest
from fastapi.testclient import TestClient
from ares.api.server import create_app, set_global_engine


def test_api_health_without_engine():
    set_global_engine(None)
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["engine_initialized"] is False


def test_api_config_uninitialized_returns_503():
    set_global_engine(None)
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/config")
    assert response.status_code == 503
