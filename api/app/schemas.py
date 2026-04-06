"""Pydantic response models matching the exact grading server key names."""

from __future__ import annotations

from pydantic import BaseModel


# ── /predict ─────────────────────────────────────────────────────────────────

class DetectionItem(BaseModel):
    label: str
    confidence: float
    bbox: list[float]


class PredictResponse(BaseModel):
    predictions: list[DetectionItem]
    model_used: str


# ── /health-status ───────────────────────────────────────────────────────────

class HealthStatusResponse(BaseModel):
    status: str
    server: str
    uptime: str


# ── /management/models ───────────────────────────────────────────────────────

class AvailableModelsResponse(BaseModel):
    available_models: list[str]


# ── /management/models/{model}/describe ──────────────────────────────────────

class ModelConfig(BaseModel):
    input_size: list[int]
    batch_size: int
    confidence_threshold: float


class ModelDescribeResponse(BaseModel):
    model: str
    config: ModelConfig
    date_registered: str


# ── /management/models/{model}/set-default ───────────────────────────────────

class SetDefaultResponse(BaseModel):
    success: bool
    default_model: str


# ── /group-info ──────────────────────────────────────────────────────────────

class GroupInfoResponse(BaseModel):
    group: str
    members: list[str]


# ── /metrics ─────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    request_rate_per_minute: float
    avg_latency_ms: float
    max_latency_ms: float
    total_requests: int
