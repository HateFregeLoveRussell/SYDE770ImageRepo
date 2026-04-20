from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..schemas import (
    AvailableModelsResponse,
    ModelConfig,
    ModelDescribeResponse,
    SetDefaultResponse,
)

router = APIRouter(prefix="/management", tags=["management"])


@router.get("/models", response_model=AvailableModelsResponse)
async def list_models(request: Request) -> AvailableModelsResponse:
    detector = request.app.state.detector
    return AvailableModelsResponse(
        available_models=detector.available_models,
    )


@router.get("/models/{model}/describe", response_model=ModelDescribeResponse)
async def describe_model(model: str, request: Request) -> ModelDescribeResponse:
    detector = request.app.state.detector
    if model not in detector.available_models:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    meta = detector.get_meta(model)
    settings = request.app.state.settings

    return ModelDescribeResponse(
        model=model,
        config=ModelConfig(
            input_size=[settings.img_size, settings.img_size],
            batch_size=settings.batch_size,
            confidence_threshold=settings.default_conf,
        ),
        date_registered=meta["date_registered"],
    )


@router.get("/models/{model}/set-default", response_model=SetDefaultResponse)
async def set_default_model(model: str, request: Request) -> SetDefaultResponse:
    detector = request.app.state.detector
    if model not in detector.available_models:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    request.app.state.settings.default_model = model
    return SetDefaultResponse(success=True, default_model=model)
