"""Tracks request metrics for both the JSON /metrics endpoint and Prometheus /prometheus."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from prometheus_client import Counter, Histogram, Gauge


# ── Prometheus counters/histograms (scraped by Prometheus for Grafana) ────────

REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total prediction requests",
    ["model_id"],
)

DETECTION_COUNT = Counter(
    "detections_total",
    "Total detections returned",
    ["model_id", "class_name"],
)

INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Inference latency in seconds",
    ["model_id"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

CONFIDENCE_HISTOGRAM = Histogram(
    "detection_confidence",
    "Confidence score distribution",
    ["model_id", "class_name"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

DETECTIONS_PER_IMAGE = Histogram(
    "detections_per_image",
    "Number of detections per image",
    ["model_id"],
    buckets=(0, 1, 2, 3, 5, 10, 20, 50),
)

AVG_CONFIDENCE = Gauge(
    "avg_confidence",
    "Rolling average confidence score",
    ["model_id"],
)

# ── Accuracy metrics (set by evaluation script) ─────────────────────────────

MODEL_PRECISION = Gauge(
    "model_precision",
    "Precision against ground truth",
    ["model_id", "class_name"],
)

MODEL_RECALL = Gauge(
    "model_recall",
    "Recall against ground truth",
    ["model_id", "class_name"],
)

MODEL_F1 = Gauge(
    "model_f1_score",
    "F1 score against ground truth",
    ["model_id", "class_name"],
)

MODEL_MAP50 = Gauge(
    "model_map50",
    "mAP50 against ground truth",
    ["model_id"],
)

MODEL_MAP50_95 = Gauge(
    "model_map50_95",
    "mAP50-95 against ground truth",
    ["model_id"],
)

MODEL_CLASS_AP50 = Gauge(
    "model_class_ap50",
    "Per-class AP50",
    ["model_id", "class_name"],
)

MODEL_CLASS_AP50_95 = Gauge(
    "model_class_ap50_95",
    "Per-class AP50-95",
    ["model_id", "class_name"],
)


# ── In-memory record buffer (for JSON /metrics endpoint) ─────────────────────

@dataclass
class RequestRecord:
    timestamp: float
    latency_ms: float
    model_id: str
    num_detections: int
    confidences: list[float] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)


class MonitoringService:
    def __init__(self, window_size: int = 200) -> None:
        self._records: deque[RequestRecord] = deque(maxlen=window_size)
        self._all_latencies: list[float] = []
        self._lock = threading.Lock()
        self._total_requests = 0
        self._start_time = time.time()

    def record(self, latency_ms: float, model_id: str,
               num_detections: int, confidences: list[float],
               class_names: list[str]) -> None:
        # ── Update Prometheus metrics ────────────────────────────────────
        REQUEST_COUNT.labels(model_id=model_id).inc()
        INFERENCE_LATENCY.labels(model_id=model_id).observe(latency_ms / 1000.0)
        DETECTIONS_PER_IMAGE.labels(model_id=model_id).observe(num_detections)

        for conf, cls in zip(confidences, class_names):
            DETECTION_COUNT.labels(model_id=model_id, class_name=cls).inc()
            CONFIDENCE_HISTOGRAM.labels(model_id=model_id, class_name=cls).observe(conf)

        if confidences:
            AVG_CONFIDENCE.labels(model_id=model_id).set(
                sum(confidences) / len(confidences)
            )

        # ── Update in-memory buffer for JSON /metrics ────────────────────
        rec = RequestRecord(
            timestamp=time.time(),
            latency_ms=latency_ms,
            model_id=model_id,
            num_detections=num_detections,
            confidences=confidences,
            class_names=class_names,
        )
        with self._lock:
            self._records.append(rec)
            self._all_latencies.append(latency_ms)
            self._total_requests += 1

    def get_metrics(self) -> dict:
        with self._lock:
            total = self._total_requests
            latencies = list(self._all_latencies)

        elapsed_minutes = max((time.time() - self._start_time) / 60.0, 1 / 60)

        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            max_lat = max(latencies)
        else:
            avg_lat = 0.0
            max_lat = 0.0

        return {
            "request_rate_per_minute": round(total / elapsed_minutes, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "total_requests": total,
        }

    @property
    def total_requests(self) -> int:
        return self._total_requests

    def get_recent_records(self, n: int | None = None) -> list[RequestRecord]:
        with self._lock:
            records = list(self._records)
        if n is not None:
            records = records[-n:]
        return records
