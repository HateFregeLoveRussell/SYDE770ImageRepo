# API Runbook

## Prerequisites

- Python 3.11+ with `.venv` activated
- Docker + Docker Compose (for Grafana analytics only)
- ngrok installed and authenticated (for grading submission)
- `pip install -r api/requirements.txt` (or deps already in `.venv`)

---

## Part A: Grading Submission Workflow

### 1. Start the API

```bash
cd api
python -m app --port 6000
# or:
../.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 6000
```

The API loads all 4 models and listens on port 6000. Default model is `yolov8s_v2`.

### 2. Verify locally

```bash
curl http://127.0.0.1:6000/health-status
# Expected: {"status":"Healthy","server":"FastAPI","uptime":"..."}
```

### 3. Start ngrok tunnel

In a **separate terminal**:

```bash
ngrok http 6000
```

Copy the HTTPS URL, e.g. `https://abc123.ngrok-free.app`.

### 4. Ping the grading server

```bash
curl -X POST -F "api_link=abc123.ngrok-free.app" http://129.97.250.133:7070/ping
```

Verify you see the correct group number.

### 5. Submit for grading

```bash
curl -X POST -F "api_link=abc123.ngrok-free.app" http://129.97.250.133:7070/submit
```

Wait for "All tests passed". Keep the API running for at least 30 minutes after submission for the inference test.

### 6. Configuration

Set your group info via environment variables before starting:

```bash
export API_GROUP_NAME="group_TODO"
export API_GROUP_MEMBERS='["Liam Prieditis"]'
cd api && python -m app --port 6000
```

Or edit `api/app/config.py` directly.

---

## Part B: Analytics / Grafana Dashboard (for Writeup)

### 1. Pull test images (if not already done)

```bash
.venv/bin/dvc pull data/augmented_dataset_v2.dvc
# or for just test images:
.venv/bin/dvc pull
```

### 2. Start the full stack via Docker Compose

```bash
cd api
docker-compose up --build
```

This starts:
- **API** on `http://localhost:6000`
- **Prometheus** on `http://localhost:9090`
- **Grafana** on `http://localhost:3000` (login: admin / admin)

### 3. Run the load test

In a separate terminal:

```bash
cd api

# Test with real images (all models, cycling)
python scripts/load_test.py \
  --images ../data/derived/yolo/test/images/ \
  --url http://127.0.0.1:6000 \
  --models yolov8s_v2 yolov8s_v2_extended

# Or with augmented dataset for more volume
python scripts/load_test.py \
  --images ../data/augmented_dataset_v2/ \
  --url http://127.0.0.1:6000 \
  --limit 200
```

### 4. View Grafana dashboard

Open `http://localhost:3000` in your browser.

- Login: `admin` / `admin`
- The "Tim Hortons Cup Detector - API Monitoring" dashboard is auto-provisioned
- Wait ~30 seconds after the load test for Prometheus to scrape and Grafana to refresh

Dashboard panels:
| Panel | What it shows |
|---|---|
| Request Throughput | Requests/second over time, by model |
| Inference Latency | p50, p95, p99 latency curves |
| Confidence Distribution | Histogram of detection confidence scores |
| Detections per Image | Average detections per image over time |
| Class Balance | Pie chart: timmies vs cup |
| Average Confidence | Gauge showing rolling mean confidence |
| HTTP Error Rate | 2xx vs 4xx/5xx over time |
| Model Comparison: Latency | Side-by-side p95 latency per model |
| Model Comparison: Detections | Total detections per model + class |

### 5. Screenshot for report

Use the Grafana "Share" button on each panel or take full-page screenshots.

---

## Part C: Quick Smoke Test (all endpoints)

```bash
# Health
curl -s http://127.0.0.1:6000/health-status | python3 -m json.tool

# Group info
curl -s http://127.0.0.1:6000/group-info | python3 -m json.tool

# List models
curl -s http://127.0.0.1:6000/management/models | python3 -m json.tool

# Describe a model
curl -s http://127.0.0.1:6000/management/models/yolov8s_v2/describe | python3 -m json.tool

# Set default model
curl -s http://127.0.0.1:6000/management/models/yolov8s_v2_extended/set-default | python3 -m json.tool

# Predict (single image)
curl -s -X POST http://127.0.0.1:6000/predict \
  -F "image=@/path/to/test.jpg;type=image/jpeg" | python3 -m json.tool

# Predict with model selection
curl -s -X POST http://127.0.0.1:6000/predict \
  -F "image=@/path/to/test.jpg;type=image/jpeg" \
  -F "model=yolov8s_v2_extended" | python3 -m json.tool

# Metrics (JSON for grading)
curl -s http://127.0.0.1:6000/metrics | python3 -m json.tool

# Prometheus metrics (for Grafana)
curl -s http://127.0.0.1:6000/prometheus | head -20

# Swagger UI
open http://127.0.0.1:6000/docs
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `Address already in use` | `lsof -ti :6000 \| xargs kill -9` |
| Models not found | Check `API_MODELS_DIR` env var or run `dvc pull models.dvc` |
| ngrok connection refused | Make sure API is on `0.0.0.0`, not `127.0.0.1` |
| Grafana shows "No data" | Wait 30s for Prometheus to scrape; check `http://localhost:9090/targets` |
| Grading server "Some tests failed" | Check JSON key names match spec exactly (see `/docs` for schemas) |
