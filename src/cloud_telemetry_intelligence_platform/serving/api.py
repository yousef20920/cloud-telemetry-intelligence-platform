"""FastAPI application for telemetry model inference."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .feature_builder import normalize_window_start
from .registry import ModelRegistry
from .schemas import AnomalyResponse, BatchResponse, InferenceWindow, RegressionResponse


def create_app(project_root: Path | None = None) -> FastAPI:
    resolved_root = (project_root or Path(os.getenv("TELEMETRY_PROJECT_ROOT", "."))).resolve()
    registry = ModelRegistry(resolved_root)
    app = FastAPI(title="Cloud Telemetry Intelligence API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        try:
            registry.ensure_loaded()
            available = True
        except FileNotFoundError:
            available = False
        return {
            "status": "ok",
            "project_root": str(resolved_root),
            "models_loaded": available,
        }

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        try:
            return registry.metrics()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/predict/anomaly", response_model=AnomalyResponse)
    def predict_anomaly(window: InferenceWindow) -> AnomalyResponse:
        try:
            predictions, preview, _ = registry.predict_anomaly(window)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return AnomalyResponse(
            service_name=window.service_name,
            host_id=window.host_id,
            window_start=normalize_window_start(window.window_start, window_minutes=window.window_minutes),
            predictions=predictions,
            feature_preview=preview,
        )

    @app.post("/predict/regression", response_model=RegressionResponse)
    def predict_regression(window: InferenceWindow) -> RegressionResponse:
        try:
            predictions, preview, _ = registry.predict_regression(window)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return RegressionResponse(
            service_name=window.service_name,
            host_id=window.host_id,
            window_start=normalize_window_start(window.window_start, window_minutes=window.window_minutes),
            predictions=predictions,
            feature_preview=preview,
        )

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(windows: list[InferenceWindow]) -> BatchResponse:
        try:
            results = registry.predict_batch(windows)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return BatchResponse(results=results)

    return app


app = create_app()

