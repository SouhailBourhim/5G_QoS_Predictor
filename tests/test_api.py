"""
Unit tests for src/deployment/api.py
Requirements: 9.2, 9.4, 9.8
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from src.deployment.api import app, SUPPORTED_SLICES, HEALTH_CRITICAL, HEALTH_WARNING

client = TestClient(app)


def _make_timesteps(n: int, near_sla: bool = False) -> list[dict]:
    """Generate n valid timestep records for eMBB."""
    base = datetime(2024, 3, 1, 0, 0, 0)
    dl = 52.0 if near_sla else 80.0
    lat = 28.0 if near_sla else 12.0
    pl  = 0.85 if near_sla else 0.1
    return [
        {
            "timestamp":     (base + timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%S"),
            "dl_throughput": dl + i * 0.01,
            "latency":       lat,
            "jitter":        1.0,
            "packet_loss":   pl,
            "prb_util":      0.92 if near_sla else 0.55,
            "active_users":  40.0,
            "reliability":   99.95,
        }
        for i in range(n)
    ]


# ─── /slices ──────────────────────────────────────────────────────────────────

def test_get_slices_returns_all():
    """GET /slices should return all 3 supported slice types."""
    resp = client.get("/slices")
    assert resp.status_code == 200
    data = resp.json()
    assert "slices" in data
    assert set(data["slices"]) == set(SUPPORTED_SLICES)


# ─── /health ──────────────────────────────────────────────────────────────────

def test_health_known_slice():
    """GET /health/eMBB should return 200 with health_status and violation_prob_30min."""
    resp = client.get("/health/eMBB")
    assert resp.status_code == 200
    data = resp.json()
    assert "health_status"        in data
    assert "violation_prob_30min" in data
    assert data["health_status"] in {"healthy", "warning", "critical"}


def test_health_unknown_slice_returns_404():
    """GET /health/{unknown} must return HTTP 404 — Req 9.4"""
    resp = client.get("/health/NonExistentSlice")
    assert resp.status_code == 404


# ─── /predict ─────────────────────────────────────────────────────────────────

def test_predict_too_few_timesteps_returns_422():
    """POST /predict with <12 timesteps must return HTTP 422 — Req 9.2"""
    payload = {
        "slice_type": "eMBB",
        "history":    _make_timesteps(6),   # only 6 — below minimum of 12
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_unknown_slice_returns_422():
    """POST /predict with unknown slice_type must return HTTP 422 (Pydantic validation)."""
    payload = {
        "slice_type": "LTE",
        "history":    _make_timesteps(24),
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_valid_request_returns_200():
    """POST /predict with valid eMBB history should return 200 with full response."""
    payload = {
        "slice_type": "eMBB",
        "history":    _make_timesteps(48),
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["slice_type"] == "eMBB"
    assert data["health_status"] in {"healthy", "warning", "critical"}
    assert len(data["horizons"]) == 3
    for h in data["horizons"]:
        assert 0.0 <= h["violation_prob"] <= 1.0
        assert h["horizon_min"] in {15, 30, 60}
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) >= 1


def test_health_status_levels():
    """health_status must be 'healthy' / 'warning' / 'critical' based on thresholds — Req 9.8"""
    payload = {
        "slice_type": "eMBB",
        "history":    _make_timesteps(48, near_sla=True),
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    prob_30 = next(h["violation_prob"] for h in data["horizons"] if h["horizon_min"] == 30)
    expected_status = (
        "critical" if prob_30 >= HEALTH_CRITICAL else
        "warning"  if prob_30 >= HEALTH_WARNING  else
        "healthy"
    )
    assert data["health_status"] == expected_status
