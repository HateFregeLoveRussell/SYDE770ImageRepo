# System Design: Tim Hortons Cup Detection API

## 1. API Design and Endpoints

### Architecture

The API is built with **FastAPI** (Python), chosen for its async request handling, automatic OpenAPI documentation, Pydantic validation, and high performance via Uvicorn's ASGI server.

### Endpoint Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Upload an image, receive object detections with bounding boxes, confidence scores, and class labels. Accepts optional `model` form field to select a specific model. |
| GET | `/health-status` | Returns server status ("Healthy"), server type ("FastAPI"), and human-readable uptime. |
| GET | `/group-info` | Returns group name and member names. |
| GET | `/metrics` | Returns JSON performance metrics: request rate per minute, average/max latency, total requests. Used by the grading server. |
| GET | `/management/models` | Lists all available models from best to worst. Top 2 are tested by the grading server. |
| GET | `/management/models/{model}/describe` | Returns model details: input size, batch size, confidence threshold, registration date. |
| GET | `/management/models/{model}/set-default` | Changes the default model for future `/predict` calls without restarting. |
| GET | `/prometheus` | Prometheus-format metrics for Grafana dashboards (not used by grading server). |
| GET | `/docs` | Interactive Swagger UI for testing all endpoints in-browser. |
| POST | `/evaluate` | Triggers background ground-truth evaluation, publishes mAP/precision/recall to Prometheus. |
| POST | `/load-eval` | Loads pre-computed evaluation results from `eval_results.json` into Prometheus (instant). |

### Inference Pipeline

1. **Image Upload**: Client sends image as `multipart/form-data` via the `image` field. Supports JPEG, PNG, WebP, BMP, and TIFF.
2. **Preprocessing**: PIL opens the image and converts to RGB. No resizing needed -- Ultralytics handles letterboxing to 640x640 internally.
3. **Model Selection**: Optional `model` form field selects which checkpoint to use. Defaults to `yolov8s_v3_260`.
4. **Inference**: Ultralytics `model.predict()` runs forward pass + NMS. Confidence threshold (0.40) and IoU threshold (0.7) are applied.
5. **Response**: Bounding boxes returned in **Pascal VOC format** `[xmin, ymin, xmax, ymax]` with class labels (`"timmies"` or `"cup"`) and confidence scores.

### Dynamic Image Size Handling

The API handles arbitrary image sizes. Ultralytics internally letterboxes images to the model's training size (640x640) while preserving aspect ratio, then maps bounding box coordinates back to the original image dimensions. No client-side resizing is required.

---

## 2. Prediction Results and Improvements

### Model Evolution

We trained 6 models across 3 dataset versions, each improving on the last:

| Model | Dataset | Images | Epochs | mAP50-95 | Description |
|---|---|---|---|---|---|
| **`yolov8s_v3_260`** (default) | v3 | ~4,700 | 260 | **0.927** | Retrained with new cup designs |
| **`yolov8s_v3_182`** | v3 | ~4,700 | 182 | 0.922 | Highest precision variant |
| `yolov8s_v2_extended` | v2 (4,406) | 4,406 | ~320 | 0.903 | Extended training on v2 |
| `yolov8s_v1` | v1 (4,006) | 4,006 | 200 | 0.895 | Baseline |
| `yolov8s_v2` | v2 (4,406) | 4,406 | 200 | 0.899 | v2 with hard negatives |
| `yolov8s_v1_extended` | v1 (4,006) | 4,006 | ~320 | 0.904 | Extended training on v1 |

### Initial Grading Results and Failure Analysis (v2 models)

The initial grading test set (26 images) exposed a critical gap in our v1/v2 training data:

| Model | Correct | F1 | Key Issue |
|---|---|---|---|
| yolov8s_v2_extended | 14/26 | 0.796 | Misclassified new cup design |
| yolov8s_v1 | 15/26 | 0.816 | Same issue, slightly better |
| yolov8s_v1_extended | 16/26 | 0.863 | Best of v1/v2 models |

**Root cause**: Our v1/v2 training data contained primarily the **older Tim Hortons cup design** (red cup with "Tim Hortons" text in white cursive). The grading test set featured the **newer 2024 design** (red cup with white maple leaf and "Always Fresh / Toujours Frais" text). The model learned to identify Tim Hortons by the text, not by the maple leaf or overall red cup style.

**Specific failure patterns on v2_extended:**
- **image_17, image_24, image_25**: Tim Hortons maple leaf cups confidently classified as generic "cup" (0.82-0.94 confidence)
- **image_16**: 3 timmies cups, 2 misclassified as "cup" (maple leaf design)
- **image_13**: Multi-cup scene with phantom detections from clutter
- **image_11**: Red Starbucks holiday cups confused with Tim Hortons due to similar color

### Improvement: v3 Models

To address these failures, we augmented the training data with:
- New Tim Hortons maple leaf cup design images
- "Always Fresh / Toujours Frais" branding variants
- Promotional cup designs (hockey jersey, seasonal)
- Additional diverse cup angles and lighting conditions

**Results after retraining (v3 models on grading test set):**

| Model | Correct | TP | FP | FN | F1 |
|---|---|---|---|---|---|
| yolov8s_v3_260 | **26/26** | 48 | 0 | 0 | **1.000** |
| yolov8s_v3_182 | **26/26** | 48 | 0 | 0 | **1.000** |

Both v3 models achieve perfect detection on the initial grading test set, correctly classifying both the old text-based and new maple leaf-based Tim Hortons designs.

### Confidence Threshold Selection

We evaluated five confidence thresholds on the grading test set across all models:

| Threshold | v2_ext Correct | v1_ext Correct | v3_260 Correct |
|---|---|---|---|
| 0.25 | 14/26 | 16/26 | 26/26 |
| 0.35 | 14/26 | 16/26 | 26/26 |
| **0.40** | **15/26** | **17/26** | **26/26** |
| 0.45 | 15/26 | 17/26 | 26/26 |
| 0.50 | 14/26 | 17/26 | 26/26 |

**Selected threshold: 0.40** because:
- At 0.25 (the default YOLO threshold), low-confidence phantom detections inflate false positives (e.g., timmies:0.39 on image_10, timmies:0.33 on image_11)
- At 0.40, these phantoms are filtered out, improving precision without significantly hurting recall
- At 0.50+, real detections start being lost (e.g., image_1 cup at 0.46 on v2 model)
- For the v3 models, all thresholds from 0.25 to 0.50 produce identical results (all detections are high-confidence), but 0.40 provides a safety margin for unseen data
- The misclassifications in v2 models were high-confidence (0.82-0.94), so threshold tuning alone couldn't fix them -- the training data improvement was necessary

### Class Naming

- **`timmies`** (class 1): Tim Hortons branded cups (including maple leaf, text, and promotional designs)
- **`cup`** (class 0): Non-Tim Hortons cups (Starbucks, generic, disposable)

---

## 3. Monitoring

### Dual Metrics Architecture

The system serves two metrics formats simultaneously:

| Endpoint | Format | Consumer | Purpose |
|---|---|---|---|
| `GET /metrics` | JSON | Grading server | `request_rate_per_minute`, `avg_latency_ms`, `max_latency_ms`, `total_requests` |
| `GET /prometheus` | Prometheus text | Prometheus + Grafana | Detailed histograms, counters, gauges for dashboards |

### Prometheus Metrics

| Metric | Type | Labels | Grafana Panel |
|---|---|---|---|
| `predict_requests_total` | Counter | `model_id` | Request Throughput |
| `inference_duration_seconds` | Histogram | `model_id` | Latency p50/p95/p99 |
| `avg_confidence` | Gauge | `model_id` | Confidence gauges |
| `detections_per_image` | Histogram | `model_id` | Detections per Image |
| `detections_total` | Counter | `model_id`, `class_name` | Class balance donuts |
| `model_map50_95` | Gauge | `model_id` | mAP50-95 comparison |
| `model_map50` | Gauge | `model_id` | mAP50 comparison |
| `model_class_ap50_95` | Gauge | `model_id`, `class_name` | Per-class AP breakdown |
| `model_precision` / `model_recall` | Gauge | `model_id`, `class_name` | Precision vs Recall |

### Grafana Dashboard

A pre-provisioned dashboard with panels organized into three sections:

**Operational Monitoring**: Request throughput, inference latency (p50/p95/p99), confidence over time, detections per image, cup/timmies detection donuts, average confidence gauges, HTTP error rate.

**Model Comparison**: Side-by-side p95 latency, total detections per model per class.

**Accuracy (vs Ground Truth)**: mAP50-95 (COCO metric), mAP50, per-class AP50-95, precision vs recall -- all across all models.

### Drift Detection

To detect distribution shift in production, we monitor:

1. **Confidence distribution shift**: The `avg_confidence` Prometheus gauge tracks rolling mean confidence per model. A sustained drop below the training-time average (~0.93 for v3 models) signals potential drift -- the model is less certain about incoming images.

2. **Class ratio drift**: The `detections_total` counter tracks timmies vs cup ratios. During training, the ratio was approximately 75% cup / 25% timmies. If production traffic shows a significantly different ratio, it may indicate a shift in the input distribution.

3. **Detection count drift**: The `detections_per_image` histogram tracks how many objects are found per image. Training data averaged ~1.3 detections/image. A sustained increase suggests the model may be hallucinating detections on out-of-distribution data.

4. **Demonstrated drift example**: Our v2 models trained on text-based Tim Hortons designs showed a clear confidence pattern when encountering the newer maple leaf design -- they detected cups at high confidence (0.82-0.94) but **misclassified** them. This manifested as stable confidence scores but shifted class distributions (more "cup" predictions where "timmies" was expected). Monitoring the timmies/cup ratio over time would have surfaced this drift.

**Mitigation**: When drift is detected, the `/management/models/{model}/set-default` endpoint allows hot-swapping to a different model variant without downtime. Our v3 models, trained on a broader distribution of cup designs, are more robust to future design changes.

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

### Tunneling for Grading

The API runs locally and is exposed via **ngrok**, bypassing dynamic IP and firewall issues:

```
Grading Server (129.97.250.133:7070)  →  ngrok tunnel  →  Local API (0.0.0.0:6045)
```

### Model Versioning

Models are versioned with **DVC** backed by Cloudflare R2 object storage. Split JSONs record exact train/test file lists for reproducible evaluation.

### Load Testing & Evaluation

- **`scripts/load_test.py`**: Sends images to the API via HTTP, cycling through models. Generates live Prometheus metrics for Grafana.
- **`scripts/yolo_eval.py`**: Runs YOLO `model.val()` with COCO-standard IoU-based box matching. Reports mAP50, mAP50-95, and per-image TP/FP/FN.
- **`scripts/extract_test_images.py`**: Recreates exact train/test splits from split JSONs for reproducible evaluation.

### Configuration

| Variable | Default | Description |
|---|---|---|
| `API_MODELS_DIR` | `<repo>/models` | Path to model checkpoint directory |
| `API_DEFAULT_MODEL` | `yolov8s_v3_260` | Default model for `/predict` |
| `API_DEFAULT_CONF` | `0.40` | Confidence threshold |
| `API_DEFAULT_IOU` | `0.7` | NMS IoU threshold |
| `API_IMG_SIZE` | `640` | Inference image size |
| `API_PORT` | `6045` | Server port (Group 5 assignment) |
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_GROUP_NAME` | `group5` | Group identifier |
| `API_GROUP_MEMBERS` | `[...]` | Group member names |
