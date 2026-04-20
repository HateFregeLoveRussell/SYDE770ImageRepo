from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import health, management, predict
from .services.detector import DetectorService
from .services.monitoring import MonitoringService

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _HAS_INSTRUMENTATOR = True
except ImportError:
    _HAS_INSTRUMENTATOR = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.start_time = time.time()
    app.state.detector = DetectorService(settings)
    app.state.monitor = MonitoringService(window_size=settings.evidently_window_size)

    loaded = app.state.detector.available_models
    print(f"Loaded {len(loaded)} model(s): {loaded}")
    print(f"Default model: {settings.default_model}")

    yield

    print("Shutting down.")


app = FastAPI(
    title="Tim Hortons Cup Detector API",
    description="YOLOv8-based object detection API for Tim Hortons cup detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(management.router)

if _HAS_INSTRUMENTATOR:
    Instrumentator().instrument(app).expose(app, endpoint="/prometheus")
