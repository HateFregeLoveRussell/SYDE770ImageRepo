from __future__ import annotations

import pathlib
import time
import threading

from fastapi import APIRouter, Request

from ..schemas import HealthStatusResponse, MetricsResponse, GroupInfoResponse
from ..services.monitoring import (
    MODEL_PRECISION, MODEL_RECALL, MODEL_F1,
    MODEL_MAP50, MODEL_MAP50_95, MODEL_CLASS_AP50, MODEL_CLASS_AP50_95,
)

router = APIRouter(tags=["status"])


def _format_uptime(seconds: float) -> str:
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs or not parts:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return ", ".join(parts)


@router.get("/health-status", response_model=HealthStatusResponse)
async def health_status(request: Request) -> HealthStatusResponse:
    uptime_sec = time.time() - request.app.state.start_time
    return HealthStatusResponse(
        status="Healthy",
        server="FastAPI",
        uptime=_format_uptime(uptime_sec),
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(request: Request) -> MetricsResponse:
    data = request.app.state.monitor.get_metrics()
    return MetricsResponse(**data)


@router.get("/group-info", response_model=GroupInfoResponse)
async def group_info(request: Request) -> GroupInfoResponse:
    settings = request.app.state.settings
    return GroupInfoResponse(
        group=settings.group_name,
        members=settings.group_members,
    )


def _run_evaluation(detector, eval_yaml: pathlib.Path) -> dict:
    """Run YOLO val() for all models and set Prometheus gauges."""
    results = {}
    for model_id in detector.available_models:
        model = detector._models[model_id]
        val_results = model.val(data=str(eval_yaml), imgsz=640, conf=0.25,
                                iou=0.7, verbose=False)
        rd = val_results.results_dict
        box = val_results.box

        map50 = float(rd.get("metrics/mAP50(B)", 0))
        map50_95 = float(rd.get("metrics/mAP50-95(B)", 0))
        precision = float(rd.get("metrics/precision(B)", 0))
        recall = float(rd.get("metrics/recall(B)", 0))
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        MODEL_MAP50.labels(model_id=model_id).set(map50)
        MODEL_MAP50_95.labels(model_id=model_id).set(map50_95)
        MODEL_PRECISION.labels(model_id=model_id, class_name="overall").set(precision)
        MODEL_RECALL.labels(model_id=model_id, class_name="overall").set(recall)
        MODEL_F1.labels(model_id=model_id, class_name="overall").set(f1)

        if hasattr(box, 'ap') and box.ap is not None:
            for i, cls_name in enumerate(val_results.names.values()):
                if i < len(box.ap):
                    cls_ap50 = float(box.ap50[i]) if hasattr(box, 'ap50') else 0
                    cls_ap = float(box.ap[i])
                    cls_p = float(box.p[i]) if hasattr(box, 'p') else 0
                    cls_r = float(box.r[i]) if hasattr(box, 'r') else 0
                    cls_f1 = 2 * cls_p * cls_r / (cls_p + cls_r + 1e-9)

                    MODEL_CLASS_AP50.labels(model_id=model_id, class_name=cls_name).set(cls_ap50)
                    MODEL_CLASS_AP50_95.labels(model_id=model_id, class_name=cls_name).set(cls_ap)
                    MODEL_PRECISION.labels(model_id=model_id, class_name=cls_name).set(cls_p)
                    MODEL_RECALL.labels(model_id=model_id, class_name=cls_name).set(cls_r)
                    MODEL_F1.labels(model_id=model_id, class_name=cls_name).set(cls_f1)

        results[model_id] = {
            "mAP50": map50, "mAP50-95": map50_95,
            "precision": precision, "recall": recall, "f1": f1,
        }

    return results


@router.post("/evaluate")
async def run_evaluation(request: Request):
    """Run ground truth evaluation on test set and publish to Prometheus gauges."""
    detector = request.app.state.detector
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    eval_yaml = repo_root / "data" / "eval_v2" / "data.yaml"

    if not eval_yaml.exists():
        return {"error": "eval data.yaml not found. Run extract_test_images.py first."}

    def bg():
        _run_evaluation(detector, eval_yaml)

    t = threading.Thread(target=bg, daemon=True)
    t.start()
    return {"message": "Evaluation started in background for all models. Check Grafana in ~5 minutes."}
