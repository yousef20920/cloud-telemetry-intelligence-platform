"""Request and response schemas for the inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TelemetryObservation(BaseModel):
    timestamp: str
    metric_name: str
    metric_value: float | None = None
    unit: str = ""
    event_type: Literal["metric", "event"] = "metric"
    event_summary: str = ""


class HistoryWindow(BaseModel):
    cpu_utilization: float | None = None
    memory_utilization: float | None = None
    packet_drop_ratio: float | None = None
    error_rate: float | None = None
    latency_ms: float | None = None
    throughput_rps: float | None = None
    request_failures: float | None = None
    event_count: int | None = None


class InferenceWindow(BaseModel):
    service_name: str
    host_id: str
    window_start: str
    window_minutes: int = Field(default=1, ge=1, le=60)
    rolling_window_size: int = Field(default=3, ge=1, le=20)
    observations: list[TelemetryObservation]
    history: list[HistoryWindow] = Field(default_factory=list)


class ModelPrediction(BaseModel):
    model_name: str
    predicted_label: int | None = None
    probability: float | None = None
    predicted_value: float | None = None
    target: str | None = None


class AnomalyResponse(BaseModel):
    service_name: str
    host_id: str
    window_start: str
    predictions: list[ModelPrediction]
    feature_preview: dict[str, float | int]


class RegressionResponse(BaseModel):
    service_name: str
    host_id: str
    window_start: str
    predictions: list[ModelPrediction]
    feature_preview: dict[str, float | int]


class BatchItemResponse(BaseModel):
    service_name: str
    host_id: str
    window_start: str
    anomaly_predictions: list[ModelPrediction]
    regression_predictions: list[ModelPrediction]


class BatchResponse(BaseModel):
    results: list[BatchItemResponse]

