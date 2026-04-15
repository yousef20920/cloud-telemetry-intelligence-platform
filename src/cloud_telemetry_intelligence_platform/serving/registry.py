"""Model loading and inference registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .feature_builder import build_feature_map, feature_vector, normalize_window_start, preview_feature_map
from .schemas import BatchItemResponse, InferenceWindow, ModelPrediction


@dataclass(slots=True)
class LoadedModel:
    name: str
    target: str
    feature_columns: list[str]
    model: Any

    def predict(self, vector: np.ndarray) -> np.ndarray:
        return self.model.predict(vector)


class ModelRegistry:
    """Load trained artifacts and run inference against aligned feature vectors."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.training_report_path = project_root / "data" / "processed" / "reports" / "training_report.json"
        self.log_path = project_root / "data" / "serving" / "predictions.jsonl"
        self.classification_models: list[LoadedModel] = []
        self.regression_models: list[LoadedModel] = []
        self.unsupervised_models: list[LoadedModel] = []
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.training_report_path.exists():
            raise FileNotFoundError(f"Training report not found at {self.training_report_path}")
        report = json.loads(self.training_report_path.read_text(encoding="utf-8"))
        self.classification_models = self._load_model_group(report.get("classification", {}), default_target="is_anomaly")
        self.regression_models = self._load_regression_models(report.get("regression", {}))
        self.unsupervised_models = self._load_model_group(report.get("unsupervised", {}), default_target="unsupervised")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._loaded = True

    def predict_anomaly(self, window: InferenceWindow) -> tuple[list[ModelPrediction], dict[str, float | int], dict[str, float]]:
        self.ensure_loaded()
        feature_columns = self._shared_feature_columns(self.classification_models)
        feature_map = build_feature_map(window, feature_columns=feature_columns)
        vector = feature_vector(feature_map, feature_columns=feature_columns)

        predictions: list[ModelPrediction] = []
        for loaded in self.classification_models:
            predicted = int(loaded.predict(vector)[0])
            probability = None
            if hasattr(loaded.model, "predict_proba"):
                probability = float(loaded.model.predict_proba(vector)[0][1])
            predictions.append(
                ModelPrediction(
                    model_name=loaded.name,
                    predicted_label=predicted,
                    probability=None if probability is None else round(probability, 6),
                    target=loaded.target,
                )
            )
        self._log("anomaly", window, predictions, feature_map)
        return predictions, preview_feature_map(feature_map), feature_map

    def predict_regression(self, window: InferenceWindow) -> tuple[list[ModelPrediction], dict[str, float | int], dict[str, float]]:
        self.ensure_loaded()
        if not self.regression_models:
            return [], {}, {}
        feature_columns = self._shared_feature_columns(self.regression_models)
        feature_map = build_feature_map(window, feature_columns=feature_columns)
        vector = feature_vector(feature_map, feature_columns=feature_columns)
        predictions: list[ModelPrediction] = []
        for loaded in self.regression_models:
            value = float(loaded.predict(vector)[0])
            predictions.append(
                ModelPrediction(
                    model_name=loaded.name,
                    predicted_value=round(value, 6),
                    target=loaded.target,
                )
            )
        self._log("regression", window, predictions, feature_map)
        return predictions, preview_feature_map(feature_map), feature_map

    def predict_batch(self, windows: list[InferenceWindow]) -> list[BatchItemResponse]:
        results: list[BatchItemResponse] = []
        for window in windows:
            anomaly_predictions, _, _ = self.predict_anomaly(window)
            regression_predictions, _, _ = self.predict_regression(window)
            results.append(
                BatchItemResponse(
                    service_name=window.service_name,
                    host_id=window.host_id,
                    window_start=normalize_window_start(window.window_start, window_minutes=window.window_minutes),
                    anomaly_predictions=anomaly_predictions,
                    regression_predictions=regression_predictions,
                )
            )
        return results

    def metrics(self) -> dict[str, object]:
        self.ensure_loaded()
        return {
            "classification_models": [asdict_model(item) for item in self.classification_models],
            "regression_models": [asdict_model(item) for item in self.regression_models],
            "unsupervised_models": [asdict_model(item) for item in self.unsupervised_models],
            "training_report_path": str(self.training_report_path),
            "prediction_log_path": str(self.log_path),
        }

    def _load_model_group(self, group: dict[str, Any], *, default_target: str) -> list[LoadedModel]:
        loaded_models: list[LoadedModel] = []
        for name, metadata in group.items():
            bundle = joblib.load(metadata["artifact_path"])
            loaded_models.append(
                LoadedModel(
                    name=name,
                    target=bundle.get("target", default_target),
                    feature_columns=list(bundle["feature_columns"]),
                    model=bundle["model"],
                )
            )
        return loaded_models

    def _load_regression_models(self, group: dict[str, Any]) -> list[LoadedModel]:
        loaded_models: list[LoadedModel] = []
        for target, models in group.items():
            for name, metadata in models.items():
                bundle = joblib.load(metadata["artifact_path"])
                loaded_models.append(
                    LoadedModel(
                        name=name,
                        target=bundle.get("target", target),
                        feature_columns=list(bundle["feature_columns"]),
                        model=bundle["model"],
                    )
                )
        return loaded_models

    @staticmethod
    def _shared_feature_columns(models: list[LoadedModel]) -> list[str]:
        if not models:
            return []
        return list(models[0].feature_columns)

    def _log(
        self,
        request_type: str,
        window: InferenceWindow,
        predictions: list[ModelPrediction],
        feature_map: dict[str, float],
    ) -> None:
        payload = {
            "request_type": request_type,
            "service_name": window.service_name,
            "host_id": window.host_id,
            "window_start": normalize_window_start(window.window_start, window_minutes=window.window_minutes),
            "prediction_count": len(predictions),
            "predictions": [item.model_dump() for item in predictions],
            "feature_preview": preview_feature_map(feature_map),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def asdict_model(model: LoadedModel) -> dict[str, object]:
    return {
        "name": model.name,
        "target": model.target,
        "feature_count": len(model.feature_columns),
    }

