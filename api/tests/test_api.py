"""End-to-end tests verifying all grading-server-required endpoints and JSON keys."""

import io
import time

import pytest
from PIL import Image
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def _make_test_image(w: int = 640, h: int = 480) -> io.BytesIO:
    img = Image.new("RGB", (w, h), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


# ── /health-status ───────────────────────────────────────────────────────────

def test_health_status_keys(client):
    resp = client.get("/health-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Healthy"
    assert data["server"] == "FastAPI"
    assert "uptime" in data
    assert isinstance(data["uptime"], str)


# ── /group-info ──────────────────────────────────────────────────────────────

def test_group_info_keys(client):
    resp = client.get("/group-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "group" in data
    assert "members" in data
    assert isinstance(data["members"], list)
    assert len(data["members"]) >= 1


# ── /management/models ───────────────────────────────────────────────────────

def test_management_models_keys(client):
    resp = client.get("/management/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "available_models" in data
    assert isinstance(data["available_models"], list)
    assert len(data["available_models"]) >= 2


# ── /management/models/{model}/describe ──────────────────────────────────────

def test_model_describe_keys(client):
    models_resp = client.get("/management/models")
    model_id = models_resp.json()["available_models"][0]
    resp = client.get(f"/management/models/{model_id}/describe")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == model_id
    assert "config" in data
    cfg = data["config"]
    assert "input_size" in cfg
    assert "batch_size" in cfg
    assert "confidence_threshold" in cfg
    assert "date_registered" in data


def test_model_describe_404(client):
    resp = client.get("/management/models/nonexistent_model_xyz/describe")
    assert resp.status_code == 404


# ── /management/models/{model}/set-default ───────────────────────────────────

def test_set_default_model(client):
    models_resp = client.get("/management/models")
    models = models_resp.json()["available_models"]
    target = models[-1]
    resp = client.get(f"/management/models/{target}/set-default")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["default_model"] == target


def test_set_default_404(client):
    resp = client.get("/management/models/nonexistent_model_xyz/set-default")
    assert resp.status_code == 404


# ── /metrics ─────────────────────────────────────────────────────────────────

def test_metrics_keys(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "request_rate_per_minute" in data
    assert "avg_latency_ms" in data
    assert "max_latency_ms" in data
    assert "total_requests" in data


# ── /predict ─────────────────────────────────────────────────────────────────

def test_predict_keys(client):
    buf = _make_test_image()
    resp = client.post(
        "/predict",
        files={"image": ("test.jpg", buf, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data, f"Missing 'predictions' key. Got: {list(data.keys())}"
    assert "model_used" in data, f"Missing 'model_used' key. Got: {list(data.keys())}"
    assert isinstance(data["predictions"], list)


def test_predict_with_model_selection(client):
    models_resp = client.get("/management/models")
    model_id = models_resp.json()["available_models"][0]
    buf = _make_test_image()
    resp = client.post(
        "/predict",
        files={"image": ("test.jpg", buf, "image/jpeg")},
        data={"model": model_id},
    )
    assert resp.status_code == 200
    assert resp.json()["model_used"] == model_id


def test_predict_png(client):
    img = Image.new("RGB", (320, 240), color=(0, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    resp = client.post(
        "/predict",
        files={"image": ("test.png", buf, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert "model_used" in data


def test_predict_detection_shape(client):
    """If there are detections, verify each has label, confidence, bbox."""
    buf = _make_test_image()
    resp = client.post(
        "/predict",
        files={"image": ("test.jpg", buf, "image/jpeg")},
    )
    data = resp.json()
    for det in data["predictions"]:
        assert "label" in det, f"Missing 'label' key in detection: {det}"
        assert "confidence" in det, f"Missing 'confidence' key in detection: {det}"
        assert "bbox" in det, f"Missing 'bbox' key in detection: {det}"
        assert isinstance(det["bbox"], list)
        assert len(det["bbox"]) == 4


# ── Metrics increment after predict ─────────────────────────────────────────

def test_metrics_increment_after_predict(client):
    before = client.get("/metrics").json()["total_requests"]
    buf = _make_test_image()
    client.post("/predict", files={"image": ("test.jpg", buf, "image/jpeg")})
    after = client.get("/metrics").json()["total_requests"]
    assert after > before
