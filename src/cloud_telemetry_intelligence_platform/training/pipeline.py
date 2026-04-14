"""Training pipeline for telemetry ML baselines."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from .metrics import classification_metrics, regression_metrics


@dataclass(slots=True)
class TrainingSummary:
    feature_path: str
    report_path: str
    model_dir: str
    feature_count: int
    row_count: int
    classification_models: list[str]
    regression_models: list[str]
    unsupervised_models: list[str]

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


class TrainingPipeline:
    """Train and compare classical ML baselines from engineered telemetry windows."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.processed_root = data_root / "data" / "processed"
        self.feature_path = self.processed_root / "features" / "window_features.csv"
        self.report_path = self.processed_root / "reports" / "training_report.json"
        self.model_dir = data_root / "artifacts" / "models"

    def train(
        self,
        *,
        feature_path: Path | None = None,
        test_size: float = 0.4,
        random_state: int = 42,
    ) -> TrainingSummary:
        self._ensure_layout()
        source_path = feature_path or self.feature_path
        rows = self._load_rows(source_path)
        if len(rows) < 4:
            raise ValueError("At least four feature rows are required for model training")

        feature_columns = self._select_feature_columns(rows)
        matrix = self._matrix_from_rows(rows, feature_columns)
        report: dict[str, Any] = {
            "row_count": len(rows),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "classification": {},
            "regression": {},
            "unsupervised": {},
        }

        report["classification"] = self._train_classifiers(
            rows,
            matrix,
            feature_columns=feature_columns,
            test_size=test_size,
            random_state=random_state,
        )
        report["regression"] = self._train_regressors(
            rows,
            feature_columns=feature_columns,
            test_size=test_size,
            random_state=random_state,
        )
        report["unsupervised"] = self._train_unsupervised(
            rows,
            matrix,
            feature_columns=feature_columns,
            random_state=random_state,
        )

        self.report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return TrainingSummary(
            feature_path=str(source_path),
            report_path=str(self.report_path),
            model_dir=str(self.model_dir),
            feature_count=len(feature_columns),
            row_count=len(rows),
            classification_models=sorted(report["classification"].keys()),
            regression_models=sorted(report["regression"].keys()),
            unsupervised_models=sorted(report["unsupervised"].keys()),
        )

    def _ensure_layout(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _select_feature_columns(self, rows: list[dict[str, str]]) -> list[str]:
        excluded = {
            "window_start",
            "window_minutes",
            "service_name",
            "host_id",
            "event_summaries",
            "target_next_latency_ms",
            "target_next_throughput_rps",
            "is_anomaly",
        }
        candidate_columns = [name for name in rows[0].keys() if name not in excluded]
        selected: list[str] = []
        for name in candidate_columns:
            try:
                for row in rows:
                    value = row.get(name, "")
                    if value == "":
                        raise ValueError
                    float(value)
            except ValueError:
                continue
            selected.append(name)
        return selected

    def _matrix_from_rows(self, rows: list[dict[str, str]], feature_columns: list[str]) -> np.ndarray:
        return np.array(
            [[float(row[column]) for column in feature_columns] for row in rows],
            dtype=float,
        )

    def _train_classifiers(
        self,
        rows: list[dict[str, str]],
        matrix: np.ndarray,
        *,
        feature_columns: list[str],
        test_size: float,
        random_state: int,
    ) -> dict[str, Any]:
        targets = np.array([int(row["is_anomaly"]) for row in rows], dtype=int)
        stratify = targets if len(set(targets)) > 1 and min(Counter(targets).values()) > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            matrix,
            targets,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        models = {
            "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            "random_forest_classifier": RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=random_state,
            ),
        }
        xgboost_cls = self._maybe_build_xgboost_classifier(random_state=random_state)
        if xgboost_cls is not None:
            models["xgboost_classifier"] = xgboost_cls

        report: dict[str, Any] = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(x_test)[:, 1]
            metrics = classification_metrics(y_test, predictions, probabilities=probabilities)
            artifact_path = self.model_dir / f"{name}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "feature_columns": feature_columns,
                    "target": "is_anomaly",
                },
                artifact_path,
            )
            report[name] = {
                "metrics": metrics,
                "artifact_path": str(artifact_path),
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
            }
        return report

    def _train_regressors(
        self,
        rows: list[dict[str, str]],
        *,
        feature_columns: list[str],
        test_size: float,
        random_state: int,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {}
        target_specs = {
            "target_next_latency_ms": {
                "linear_regression": LinearRegression(),
                "random_forest_regressor": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=8,
                    random_state=random_state,
                ),
            },
            "target_next_throughput_rps": {
                "linear_regression": LinearRegression(),
                "random_forest_regressor": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=8,
                    random_state=random_state,
                ),
            },
        }

        xgboost_reg = self._maybe_build_xgboost_regressor(random_state=random_state)
        if xgboost_reg is not None:
            target_specs["target_next_latency_ms"]["xgboost_regressor"] = xgboost_reg
            target_specs["target_next_throughput_rps"]["xgboost_regressor"] = self._maybe_build_xgboost_regressor(
                random_state=random_state
            )

        for target_name, models in target_specs.items():
            valid_rows = [row for row in rows if row.get(target_name, "") != ""]
            if len(valid_rows) < 4:
                continue
            matrix = self._matrix_from_rows(valid_rows, feature_columns)
            targets = np.array([float(row[target_name]) for row in valid_rows], dtype=float)
            x_train, x_test, y_train, y_test = train_test_split(
                matrix,
                targets,
                test_size=test_size,
                random_state=random_state,
            )
            per_target: dict[str, Any] = {}
            for name, model in models.items():
                model.fit(x_train, y_train)
                predictions = model.predict(x_test)
                metrics = regression_metrics(y_test, predictions)
                artifact_path = self.model_dir / f"{name}_{target_name}.joblib"
                joblib.dump(
                    {
                        "model": model,
                        "feature_columns": feature_columns,
                        "target": target_name,
                    },
                    artifact_path,
                )
                per_target[name] = {
                    "metrics": metrics,
                    "artifact_path": str(artifact_path),
                    "train_rows": int(len(x_train)),
                    "test_rows": int(len(x_test)),
                }
            report[target_name] = per_target
        return report

    def _train_unsupervised(
        self,
        rows: list[dict[str, str]],
        matrix: np.ndarray,
        *,
        feature_columns: list[str],
        random_state: int,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {}
        labels = np.array([int(row["is_anomaly"]) for row in rows], dtype=int)

        isolation_forest = IsolationForest(random_state=random_state, contamination="auto")
        isolation_forest.fit(matrix)
        raw_predictions = isolation_forest.predict(matrix)
        anomaly_predictions = np.where(raw_predictions == -1, 1, 0)
        iso_metrics = classification_metrics(labels, anomaly_predictions)
        iso_artifact = self.model_dir / "isolation_forest.joblib"
        joblib.dump(
            {
                "model": isolation_forest,
                "feature_columns": feature_columns,
            },
            iso_artifact,
        )
        report["isolation_forest"] = {
            "metrics": iso_metrics,
            "artifact_path": str(iso_artifact),
            "predicted_anomalies": int(anomaly_predictions.sum()),
        }

        cluster_count = min(3, len(rows) - 1)
        if cluster_count >= 2:
            kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=random_state)
            cluster_labels = kmeans.fit_predict(matrix)
            silhouette = float(silhouette_score(matrix, cluster_labels)) if len(rows) > cluster_count else None
            cluster_sizes = Counter(int(label) for label in cluster_labels)
            anomaly_by_cluster = {
                str(cluster): {
                    "size": int(size),
                    "anomaly_rate": float(
                        sum(labels[index] for index, value in enumerate(cluster_labels) if int(value) == cluster) / size
                    ),
                }
                for cluster, size in cluster_sizes.items()
            }
            kmeans_artifact = self.model_dir / "kmeans.joblib"
            joblib.dump(
                {
                    "model": kmeans,
                    "feature_columns": feature_columns,
                },
                kmeans_artifact,
            )
            report["kmeans"] = {
                "artifact_path": str(kmeans_artifact),
                "cluster_count": cluster_count,
                "silhouette_score": silhouette,
                "cluster_summary": anomaly_by_cluster,
            }

        return report

    @staticmethod
    def _maybe_build_xgboost_classifier(*, random_state: int):
        try:
            from xgboost import XGBClassifier
        except Exception:
            return None
        return XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )

    @staticmethod
    def _maybe_build_xgboost_regressor(*, random_state: int):
        try:
            from xgboost import XGBRegressor
        except Exception:
            return None
        return XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )

