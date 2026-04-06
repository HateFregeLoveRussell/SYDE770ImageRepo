# System Design: Tim Hortons Cup Detection API

## 1. API and Detection Performance

### Architecture

The API is built with **FastAPI** (Python), chosen for its async request handling, automatic OpenAPI documentation, Pydantic validation, and high performance via Uvicorn's ASGI server.

```
Client Request
    │
    ▼
┌──────────────────────────────────────┐
│  FastAPI Application (port 6045)     │
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
3. **Model Selection**: Optional `model` form field selects which checkpoint to use. Defaults to `yolov8s_v2_extended` (configurable at runtime via `/management/models/{model}/set-default`).
4. **Inference**: Ultralytics `model.predict()` runs forward pass + NMS. Confidence threshold (0.25) and IoU threshold (0.7) are applied.
5. **Response**: Bounding boxes returned in **Pascal VOC format** `[xmin, ymin, xmax, ymax]` with class labels (`"timmies"` or `"cup"`) and confidence scores.

### Dynamic Image Size Handling

The API handles arbitrary image sizes. Ultralytics internally letterboxes images to the model's training size (640x640) while preserving aspect ratio, then maps bounding box coordinates back to the original image dimensions. No client-side resizing is required.

### Multi-Model Support

All `.pt` checkpoints in the `models/` directory are loaded at startup into memory (~21 MB each, ~85 MB total for 4 models). Model selection is per-request with zero cold-start latency. Available models are listed at `/management/models`, ordered to present diverse model characteristics to the grading server.

---

## 2. Prediction Results

### Model Selection Rationale

Four YOLOv8s (Small, 11.1M parameters) checkpoints are available, all trained with the best hyperparameters found via 3-phase Optuna Bayesian search (56 trials across architecture, optimizer, and loss weight phases):

| Model | Dataset | Epochs | Description |
|---|---|---|---|
| **`yolov8s_v2_extended`** (default) | Augmented v2 (4,406 images) | 1000 (early stopped) | Extended training on v2 with hard negatives |
| `yolov8s_v1` | Augmented v1 (4,006 images) | 200 (patience=50) | Standard training on v1 |
| `yolov8s_v2` | Augmented v2 (4,406 images) | 200 (patience=50) | Standard training on v2 with hard negatives |
| `yolov8s_v1_extended` | Augmented v1 (4,006 images) | 1000 (early stopped) | Extended training on v1 |

The model ordering (best-to-worst for grading) intentionally alternates extended and standard models to provide diverse detection characteristics across the top 2 tested models.

**`yolov8s_v2_extended`** is the default because:
- Trained on the latest, most diverse augmented dataset (v2) including 400 hard negative images
- Extended training for more epochs allows deeper convergence
- Best config: YOLOv8s, 640px, 10 frozen layers, AdamW (lr=7.7e-4), custom loss weights (box=0.115, cls=0.335, dfl=1.479)

### Test Set Evaluation (v2 held-out test set, 882 images)

All models were evaluated on the v2 held-out test set using the exact train/test split from training (80/20 stratified split, seed=42):

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | TP | FP | FN | Perfect Images |
|---|---|---|---|---|---|---|---|---|---|
| **yolov8s_v2_extended** | 0.977 | 0.904 | 0.971 | 0.957 | 0.964 | 1137 | 58 | 41 | 826/882 (93.7%) |
| yolov8s_v2 | 0.976 | 0.899 | 0.980 | 0.944 | 0.962 | 1141 | 63 | 37 | 818/882 (92.7%) |
| yolov8s_v1_extended | 0.988 | 0.945 | 0.987 | 0.977 | 0.982 | 1160 | 48 | 18 | 833/882 (94.4%) |
| yolov8s_v1 | 0.990 | 0.936 | 0.991 | 0.976 | 0.983 | 1156 | 57 | 22 | 823/882 (93.3%) |

Key findings:
- All models achieve >97% mAP50 and >89% mAP50-95 on the held-out test set
- v1 models score higher on v2 test data because the v2 test set includes 80 hard negative images (no cups) -- v1 models were not trained on hard negatives and are more conservative, producing fewer false positives
- Extended training consistently improves mAP50-95 over standard training (+0.005 to +0.009)
- The default model (`yolov8s_v2_extended`) correctly identifies 93.7% of test images with exact detection count matches

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
  "model_used": "yolov8s_v2_extended"
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

Rich, time-series metrics scraped by Prometheus every 5 seconds:

| Metric | Type | Labels | Grafana Panel |
|---|---|---|---|
| `predict_requests_total` | Counter | `model_id` | Request Throughput |
| `inference_duration_seconds` | Histogram | `model_id` | Latency p50/p95/p99 |
| `avg_confidence` | Gauge | `model_id` | Confidence over time / Avg gauges |
| `detections_per_image` | Histogram | `model_id` | Detections per Image |
| `detections_total` | Counter | `model_id`, `class_name` | Cup/Timmies donuts |
| `model_map50_95` | Gauge | `model_id` | mAP50-95 bar gauge |
| `model_map50` | Gauge | `model_id` | mAP50 bar gauge |
| `model_class_ap50_95` | Gauge | `model_id`, `class_name` | Per-Class AP50-95 |
| `model_precision` | Gauge | `model_id`, `class_name` | Precision vs Recall |
| `model_recall` | Gauge | `model_id`, `class_name` | Precision vs Recall |
| `http_requests_total` | Counter | `status`, `handler` | HTTP Error Rate |

Additionally, `prometheus_fastapi_instrumentator` automatically tracks HTTP-level metrics (request duration, status codes, in-progress requests).

### Grafana Dashboard

A pre-provisioned dashboard ("Tim Hortons Cup Detector - API Monitoring") with 13 panels across three sections:

**Operational Monitoring:**
1. **Request Throughput** -- time series of req/s by model
2. **Inference Latency** -- p50, p95, p99 percentile curves
3. **Confidence Score Distribution** -- rolling average confidence per model over time
4. **Detections per Image** -- rolling average of detections per image
5. **Cup Detections by Model** -- donut chart of cup detections per model
6. **Timmies Detections by Model** -- donut chart of timmies detections per model
7. **Average Confidence** -- gauge with color-coded thresholds per model
8. **HTTP Error Rate** -- 2xx vs 4xx/5xx over time

**Model Comparison:**
9. **Model Comparison: Latency** -- side-by-side p95 latency bar gauge
10. **Model Comparison: Total Detections** -- total detections per model per class

**Accuracy (vs Ground Truth):**
11. **Model Accuracy: mAP50-95** -- COCO-standard mAP for all 4 models
12. **Model Accuracy: mAP50** -- mAP at IoU=0.50 for all 4 models
13. **Per-Class AP50-95** -- cup vs timmies accuracy breakdown per model
14. **Precision vs Recall** -- all 4 models compared

---

## 4. Implementation Details

### Deployment Architecture

```
┌─────────────────────────────────────────────┐
│  Docker Compose                              │
│                                              │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐ │
│  │ FastAPI   │  │ Prometheus │  │ Grafana  │ │
│  │ :6045    │──│ :9090      │──│ :3000    │ │
│  │          │  │ scrapes    │  │ queries  │ │
│  │ models/  │  │ /prometheus│  │ prom DB  │ │
│  │ (ro vol) │  │ every 5s   │  │          │ │
│  └──────────┘  └────────────┘  └──────────┘ │
└─────────────────────────────────────────────┘
```

Three Docker Compose services:
- **api**: FastAPI + Uvicorn, model weights mounted as read-only volume
- **prometheus**: Scrapes `/prometheus` endpoint every 5 seconds
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
Local API (0.0.0.0:6045)
```

The API binds to `0.0.0.0` (all interfaces) so ngrok can forward traffic. This bypasses DHCP dynamic IP issues and university firewall restrictions.

### Model Versioning

Models are versioned with **DVC** (Data Version Control) backed by Cloudflare R2 object storage:
- `models.dvc` tracks 4 checkpoint files + split metadata (~90 MB total)
- `dvc pull models.dvc` materializes the weights locally
- Split JSONs (`v2_split_actual.json`, `v2_split_corrected.json`) record exact train/test file lists for reproducible evaluation

### Dynamic Default Model

The `/management/models/{model}/set-default` endpoint allows changing the default model at runtime without restarting the server. This is stored in the in-memory settings and takes effect immediately for subsequent `/predict` requests.

### Load Testing & Evaluation

Two evaluation workflows:
- **`scripts/load_test.py`**: Sends images from a directory to the API via HTTP, cycling through models. Generates live Prometheus metrics for Grafana visualization. Reports latency percentiles, detection counts, and throughput.
- **`scripts/yolo_eval.py`**: Runs proper YOLO `model.val()` on the held-out test set using COCO-standard IoU-based box matching. Reports mAP50, mAP50-95, precision, recall, and per-image TP/FP/FN counts for all 4 models.

### Configuration

All settings are configurable via environment variables with the `API_` prefix:

| Variable | Default | Description |
|---|---|---|
| `API_MODELS_DIR` | `<repo>/models` | Path to model checkpoint directory |
| `API_DEFAULT_MODEL` | `yolov8s_v2_extended` | Default model for `/predict` |
| `API_DEFAULT_CONF` | `0.25` | Confidence threshold |
| `API_DEFAULT_IOU` | `0.7` | NMS IoU threshold |
| `API_IMG_SIZE` | `640` | Inference image size |
| `API_PORT` | `6045` | Server port (Group 5 assignment) |
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_GROUP_NAME` | `group5` | Group identifier for `/group-info` |
| `API_GROUP_MEMBERS` | `[...]` | Group member names |
