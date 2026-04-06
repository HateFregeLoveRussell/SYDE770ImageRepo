# System Design: Tim Hortons Cup Detection API

## 1. API and Detection Performance

### Architecture

The API is built with **FastAPI** (Python), chosen for its async request handling, automatic OpenAPI documentation, Pydantic validation, and high performance via Uvicorn's ASGI server.

```
Client Request
    │
    ▼
┌──────────────────────────────────────┐
│  FastAPI Application (port 6000)     │
│  ├── /predict          (POST)        │
│  ├── /health-status    (GET)         │
│  ├── /group-info       (GET)         │
│  ├── /metrics          (GET, JSON)   │
│  ├── /prometheus       (GET, Prom)   │
│  ├── /management/models              │
│  │   ├── /            (GET, list)    │
│  │   ├── /{m}/describe (GET)         │
│  │   └── /{m}/set-default (GET)      │
│  └── /docs             (Swagger UI)  │
│                                      │
│  Services:                           │
│  ├── DetectorService (YOLO models)   │
│  └── MonitoringService (metrics)     │
└──────────────────────────────────────┘
```

### Inference Pipeline

1. **Image Upload**: Client sends image as `multipart/form-data` via the `image` field. Supports JPEG, PNG, WebP, BMP, and TIFF.
2. **Preprocessing**: PIL opens the image and converts to RGB. No resizing needed -- Ultralytics handles letterboxing to the model's input size (640x640) internally.
3. **Model Selection**: Optional `model` form field selects which checkpoint to use. Defaults to `yolov8s_v2` (configurable at runtime via `/management/models/{model}/set-default`).
4. **Inference**: Ultralytics `model.predict()` runs forward pass + NMS. Confidence threshold (0.25) and IoU threshold (0.7) are applied.
5. **Response**: Bounding boxes returned in **Pascal VOC format** `[xmin, ymin, xmax, ymax]` with class labels (`"timmies"` or `"cup"`) and confidence scores.

### Dynamic Image Size Handling

The API handles arbitrary image sizes. Ultralytics internally letterboxes images to the model's training size (640x640) while preserving aspect ratio, then maps bounding box coordinates back to the original image dimensions. No client-side resizing is required.

### Multi-Model Support

All `.pt` checkpoints in the `models/` directory are loaded at startup into memory (~21 MB each, ~85 MB total for 4 models). Model selection is per-request with zero cold-start latency. Available models are listed at `/management/models`, sorted with the default (best) model first.

---

## 2. Prediction Results

### Model Selection Rationale

Four YOLOv8s (Small, 11.1M parameters) checkpoints are available:

| Model | Dataset | Epochs | mAP50-95 | Rationale |
|---|---|---|---|---|
| **`yolov8s_v2`** (default) | Augmented v2 (4,409 images) | 200 (patience=50) | 0.897 | Best Optuna-tuned config on latest dataset |
| `yolov8s_v2_extended` | Augmented v2 (4,409 images) | 1000 (early stopped) | 0.904 | Extended training on v2 |
| `yolov8s_v1` | Augmented v1 (4,009 images) | 200 (patience=50) | 0.895 | Baseline on earlier dataset |
| `yolov8s_v1_extended` | Augmented v1 (4,009 images) | 1000 (early stopped) | 0.904 | Extended training on v1 |

**`yolov8s_v2`** is the default because:
- Trained on the latest, most diverse augmented dataset (v2)
- Uses the best hyperparameters found by a 3-phase Optuna Bayesian search (56 trials across architecture, optimizer, and loss weight phases)
- Best config: YOLOv8s, 640px, 10 frozen layers, AdamW (lr=7.7e-4), custom loss weights (box=0.115, cls=0.335, dfl=1.479)

### Class Naming

- **`timmies`** (class 1): Tim Hortons branded cups
- **`cup`** (class 0): Non-Tim Hortons cups (other brands, generic cups)

Originally named `TimHortonsCup` and `NonTimHortonsCup` during training. Renamed in model checkpoint metadata and all codebase references without retraining (YOLO labels use numeric class IDs, so the weights are unaffected).

### Detection Output Format

```json
{
  "predictions": [
    {
      "label": "timmies",
      "confidence": 0.91,
      "bbox": [42.0, 58.0, 214.0, 368.0]
    }
  ],
  "model_used": "yolov8s_v2"
}
```

Bounding boxes are in Pascal VOC format: `[xmin, ymin, xmax, ymax]` in pixel coordinates relative to the original image dimensions.

---

## 3. Monitoring

### Dual Metrics Architecture

The system serves two metrics formats simultaneously to satisfy both grading requirements and operational monitoring:

| Endpoint | Format | Consumer | Purpose |
|---|---|---|---|
| `GET /metrics` | JSON | Grading server | `request_rate_per_minute`, `avg_latency_ms`, `max_latency_ms`, `total_requests` |
| `GET /prometheus` | Prometheus text | Prometheus + Grafana | Detailed histograms, counters, gauges |

### JSON Metrics (`/metrics`)

Computed from an in-memory buffer tracking all requests since startup:
- **`request_rate_per_minute`**: Total requests / elapsed minutes
- **`avg_latency_ms`**: Mean inference latency across all requests
- **`max_latency_ms`**: Maximum observed inference latency
- **`total_requests`**: Cumulative request count

### Prometheus Metrics (`/prometheus`)

Rich, time-series metrics scraped by Prometheus every 15 seconds:

| Metric | Type | Labels | Grafana Panel |
|---|---|---|---|
| `predict_requests_total` | Counter | `model_id` | Request Throughput |
| `inference_duration_seconds` | Histogram | `model_id` | Latency p50/p95/p99 |
| `detection_confidence` | Histogram | `model_id`, `class_name` | Confidence Distribution |
| `detections_per_image` | Histogram | `model_id` | Detections per Image |
| `detections_total` | Counter | `model_id`, `class_name` | Class Balance (pie) |
| `avg_confidence` | Gauge | `model_id` | Avg Confidence (gauge) |
| `http_requests_total` | Counter | `status`, `handler` | HTTP Error Rate |

Additionally, `prometheus_fastapi_instrumentator` automatically tracks HTTP-level metrics (request duration, status codes, in-progress requests).

### Grafana Dashboard

A pre-provisioned dashboard ("Tim Hortons Cup Detector - API Monitoring") with 9 panels:

1. **Request Throughput** -- time series of req/s by model
2. **Inference Latency** -- p50, p95, p99 percentile curves
3. **Confidence Score Distribution** -- histogram of detection confidence values
4. **Detections per Image** -- rolling average of detections per image
5. **Detection Class Balance** -- donut chart showing timmies vs cup ratio
6. **Average Confidence** -- gauge with color-coded thresholds (red < 0.4, orange < 0.6, yellow < 0.8, green >= 0.8)
7. **HTTP Error Rate** -- 2xx vs 4xx/5xx over time
8. **Model Comparison: Latency** -- side-by-side p95 latency bar gauge
9. **Model Comparison: Detections** -- total detections per model per class

---

## 4. Implementation Details

### Deployment Architecture

```
┌─────────────────────────────────────────────┐
│  Docker Compose                              │
│                                              │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐ │
│  │ FastAPI   │  │ Prometheus │  │ Grafana  │ │
│  │ :6000    │──│ :9090      │──│ :3000    │ │
│  │          │  │ scrapes    │  │ queries  │ │
│  │ models/  │  │ /prometheus│  │ prom DB  │ │
│  │ (ro vol) │  │ every 15s  │  │          │ │
│  └──────────┘  └────────────┘  └──────────┘ │
└─────────────────────────────────────────────┘
```

Three Docker Compose services:
- **api**: FastAPI + Uvicorn, model weights mounted as read-only volume
- **prometheus**: Scrapes `/prometheus` endpoint every 15 seconds
- **grafana**: Pre-provisioned datasource + dashboard, accessible at port 3000

### Tunneling for Grading

For the grading submission, the API runs locally (without Docker) and is exposed via **ngrok**:

```
Grading Server (129.97.250.133:7070)
        │
        ▼
ngrok tunnel (abc123.ngrok-free.app)
        │
        ▼
Local API (0.0.0.0:6000)
```

The API binds to `0.0.0.0` (all interfaces) so ngrok can forward traffic. This bypasses DHCP dynamic IP issues and university firewall restrictions.

### Model Versioning

Models are versioned with **DVC** (Data Version Control) backed by Cloudflare R2 object storage:
- `models.dvc` tracks 4 checkpoint files (~90 MB total)
- `dvc pull models.dvc` materializes the weights locally
- `dvc push models.dvc` pushes updated weights after class rename

### Dynamic Default Model

The `/management/models/{model}/set-default` endpoint allows changing the default model at runtime without restarting the server. This is stored in the in-memory settings and takes effect immediately for subsequent `/predict` requests.

### Load Testing

`scripts/load_test.py` sends images from a specified directory to the API, cycling through models if multiple are specified. It prints per-image results and a final summary with latency percentiles, detection counts, and throughput. After running, the Prometheus + Grafana stack has rich data for dashboard screenshots.

### Configuration

All settings are configurable via environment variables with the `API_` prefix:

| Variable | Default | Description |
|---|---|---|
| `API_MODELS_DIR` | `<repo>/models` | Path to model checkpoint directory |
| `API_DEFAULT_MODEL` | `yolov8s_v2` | Default model for `/predict` |
| `API_DEFAULT_CONF` | `0.25` | Confidence threshold |
| `API_DEFAULT_IOU` | `0.7` | NMS IoU threshold |
| `API_IMG_SIZE` | `640` | Inference image size |
| `API_PORT` | `6000` | Server port |
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_GROUP_NAME` | `group_TODO` | Group identifier for `/group-info` |
| `API_GROUP_MEMBERS` | `["Liam Prieditis"]` | Group member names |
