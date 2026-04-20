from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from ..schemas import PredictResponse, DetectionItem

router = APIRouter(tags=["inference"])

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)"),
    model: str = Form(None, description="Model ID (omit for default)"),
) -> PredictResponse:
    detector = request.app.state.detector
    monitor = request.app.state.monitor
    settings = request.app.state.settings

    model_id = model or settings.default_model
    if model_id not in detector.available_models:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model_id}'. Available: {detector.available_models}",
        )

    try:
        img = Image.open(image.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}") from exc

    t0 = time.perf_counter()
    results = detector.run_inference(img, model_id=model_id)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    predictions: list[DetectionItem] = []
    confidences: list[float] = []
    class_names: list[str] = []

    for det in results:
        predictions.append(DetectionItem(
            label=det["label"],
            confidence=round(det["confidence"], 4),
            bbox=det["bbox"],
        ))
        confidences.append(det["confidence"])
        class_names.append(det["label"])

    monitor.record(
        latency_ms=latency_ms,
        model_id=model_id,
        num_detections=len(predictions),
        confidences=confidences,
        class_names=class_names,
    )

    return PredictResponse(
        predictions=predictions,
        model_used=model_id,
    )
