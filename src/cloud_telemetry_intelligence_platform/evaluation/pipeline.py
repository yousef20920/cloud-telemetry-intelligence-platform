"""Evaluation pipeline for telemetry ML baselines."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
)

from ..training.metrics import classification_metrics, regression_metrics
from .operational import benchmark_predict, feature_groups
from .rendering import render_html_dashboard, render_markdown_dashboard, render_svg_bar_chart


@dataclass(slots=True)
class EvaluationSummary:
    feature_path: str
    training_report_path: str
    report_path: str
    dashboard_markdown_path: str
    dashboard_html_path: str
    classification_models: list[str]
    regression_models: list[str]
    unsupervised_models: list[str]

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


class EvaluationPipeline:
    """Generate comparison reports, operational metrics, and dashboards for telemetry models."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.processed_root = data_root / "data" / "processed"
        self.feature_path = self.processed_root / "features" / "window_features.csv"
        self.training_report_path = self.processed_root / "reports" / "training_report.json"
        self.report_path = self.processed_root / "reports" / "evaluation_report.json"
        self.dashboard_markdown_path = self.processed_root / "reports" / "evaluation_dashboard.md"
        self.dashboard_html_path = self.processed_root / "reports" / "evaluation_dashboard.html"
        self.threshold_sweep_path = self.processed_root / "reports" / "threshold_sweeps.csv"
        self.classification_error_path = self.processed_root / "reports" / "classification_error_analysis.csv"
        self.regression_error_path = self.processed_root / "reports" / "regression_error_analysis.csv"
        self.ablation_path = self.processed_root / "reports" / "feature_ablation.csv"
        self.importance_dir = self.processed_root / "reports" / "feature_importance"

    def evaluate(
        self,
        *,
        feature_path: Path | None = None,
        training_report_path: Path | None = None,
    ) -> EvaluationSummary:
        self._ensure_layout()
        features_source = feature_path or self.feature_path
        training_report_source = training_report_path or self.training_report_path
        rows = self._load_rows(features_source)
        training_report = json.loads(training_report_source.read_text(encoding="utf-8"))
        feature_columns = list(training_report["feature_columns"])
        matrix = self._matrix_from_rows(rows, feature_columns)
        groups = feature_groups(feature_columns)

        evaluation_report: dict[str, Any] = {
            "row_count": len(rows),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "classification": {},
            "regression": {},
            "unsupervised": {},
            "artifacts": {
                "threshold_sweeps_csv": str(self.threshold_sweep_path),
                "classification_error_csv": str(self.classification_error_path),
                "regression_error_csv": str(self.regression_error_path),
                "feature_ablation_csv": str(self.ablation_path),
                "dashboard_markdown": str(self.dashboard_markdown_path),
                "dashboard_html": str(self.dashboard_html_path),
            },
        }

        threshold_rows: list[dict[str, object]] = []
        classification_error_rows: list[dict[str, object]] = []
        regression_error_rows: list[dict[str, object]] = []
        ablation_rows: list[dict[str, object]] = []
        feature_svg_paths: list[str] = []

        classification_rows = self._evaluate_classification_models(
            rows=rows,
            matrix=matrix,
            training_report=training_report["classification"],
            groups=groups,
            threshold_rows=threshold_rows,
            error_rows=classification_error_rows,
            ablation_rows=ablation_rows,
            feature_svg_paths=feature_svg_paths,
        )
        for row in classification_rows:
            evaluation_report["classification"][row["model_name"]] = row

        regression_rows = self._evaluate_regression_models(
            rows=rows,
            training_report=training_report["regression"],
            groups=groups,
            error_rows=regression_error_rows,
            ablation_rows=ablation_rows,
            feature_svg_paths=feature_svg_paths,
        )
        for row in regression_rows:
            evaluation_report["regression"].setdefault(row["target"], {})[row["model_name"]] = row

        unsupervised_rows = self._evaluate_unsupervised_models(
            rows=rows,
            matrix=matrix,
            training_report=training_report["unsupervised"],
        )
        for row in unsupervised_rows:
            evaluation_report["unsupervised"][row["model_name"]] = row

        self._write_csv(self.threshold_sweep_path, threshold_rows)
        self._write_csv(self.classification_error_path, classification_error_rows)
        self._write_csv(self.regression_error_path, regression_error_rows)
        self._write_csv(self.ablation_path, ablation_rows)

        markdown_dashboard = render_markdown_dashboard(
            classification_rows=classification_rows,
            regression_rows=regression_rows,
            unsupervised_rows=unsupervised_rows,
        )
        self.dashboard_markdown_path.write_text(markdown_dashboard, encoding="utf-8")
        summary_block = {
            "row_count": len(rows),
            "feature_count": len(feature_columns),
            "classification_count": len(classification_rows),
            "regression_count": len(regression_rows),
            "unsupervised_count": len(unsupervised_rows),
        }
        html_dashboard = render_html_dashboard(
            title="Telemetry Model Dashboard",
            summary=summary_block,
            markdown_dashboard=markdown_dashboard,
            feature_svg_paths=feature_svg_paths,
        )
        self.dashboard_html_path.write_text(html_dashboard, encoding="utf-8")

        self.report_path.write_text(json.dumps(evaluation_report, indent=2, sort_keys=True), encoding="utf-8")
        return EvaluationSummary(
            feature_path=str(features_source),
            training_report_path=str(training_report_source),
            report_path=str(self.report_path),
            dashboard_markdown_path=str(self.dashboard_markdown_path),
            dashboard_html_path=str(self.dashboard_html_path),
            classification_models=[row["model_name"] for row in classification_rows],
            regression_models=[f'{row["target"]}:{row["model_name"]}' for row in regression_rows],
            unsupervised_models=[row["model_name"] for row in unsupervised_rows],
        )

    def _ensure_layout(self) -> None:
        for directory in (
            self.report_path.parent,
            self.importance_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _load_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _matrix_from_rows(rows: list[dict[str, str]], feature_columns: list[str]) -> np.ndarray:
        return np.array(
            [[float(row[column]) for column in feature_columns] for row in rows],
            dtype=float,
        )

    def _evaluate_classification_models(
        self,
        *,
        rows: list[dict[str, str]],
        matrix: np.ndarray,
        training_report: dict[str, Any],
        groups: dict[str, list[int]],
        threshold_rows: list[dict[str, object]],
        error_rows: list[dict[str, object]],
        ablation_rows: list[dict[str, object]],
        feature_svg_paths: list[str],
    ) -> list[dict[str, Any]]:
        labels = np.array([int(row["is_anomaly"]) for row in rows], dtype=int)
        service_names = [row["service_name"] for row in rows]
        results: list[dict[str, Any]] = []

        for model_name, metadata in training_report.items():
            bundle = joblib.load(metadata["artifact_path"])
            model = bundle["model"]
            feature_columns = bundle["feature_columns"]
            predictions = model.predict(matrix)
            probabilities = model.predict_proba(matrix)[:, 1] if hasattr(model, "predict_proba") else None
            metrics = classification_metrics(labels, predictions, probabilities=probabilities)
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            operational = benchmark_predict(lambda: model.predict(matrix))
            operational["throughput_rows_per_sec"] = round(
                len(rows) / max(operational["batch_latency_ms_mean"] / 1000.0, 1e-9),
                3,
            )
            operational["false_positive_rate"] = round(
                float(fp / max(fp + tn, 1)),
                6,
            )
            stability = self._service_accuracy_breakdown(service_names, labels, predictions)

            results.append(
                {
                    "model_name": model_name,
                    **{key: (round(value, 6) if isinstance(value, float) else value) for key, value in metrics.items()},
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                    **operational,
                    "service_stability": stability,
                }
            )

            self._append_threshold_sweeps(
                model_name=model_name,
                labels=labels,
                probabilities=probabilities,
                threshold_rows=threshold_rows,
            )
            self._append_classification_errors(
                rows=rows,
                model_name=model_name,
                labels=labels,
                predictions=predictions,
                probabilities=probabilities,
                error_rows=error_rows,
            )
            baseline_score = float(metrics["f1"])
            for group_name, indices in groups.items():
                ablated_matrix = matrix.copy()
                ablated_matrix[:, indices] = 0.0
                ablated_predictions = model.predict(ablated_matrix)
                ablated_score = float(classification_metrics(labels, ablated_predictions)["f1"])
                ablation_rows.append(
                    {
                        "model_name": model_name,
                        "target": "is_anomaly",
                        "group_name": group_name,
                        "baseline_metric": baseline_score,
                        "ablated_metric": ablated_score,
                        "delta": round(ablated_score - baseline_score, 6),
                    }
                )

            svg_path = self._write_importance_chart(model_name, model, feature_columns)
            if svg_path:
                feature_svg_paths.append(str(svg_path))

        return results

    def _evaluate_regression_models(
        self,
        *,
        rows: list[dict[str, str]],
        training_report: dict[str, Any],
        groups: dict[str, list[int]],
        error_rows: list[dict[str, object]],
        ablation_rows: list[dict[str, object]],
        feature_svg_paths: list[str],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for target_name, models in training_report.items():
            valid_rows = [row for row in rows if row.get(target_name, "") != ""]
            if not valid_rows:
                continue
            any_bundle = joblib.load(next(iter(models.values()))["artifact_path"])
            feature_columns = any_bundle["feature_columns"]
            matrix = self._matrix_from_rows(valid_rows, feature_columns)
            labels = np.array([float(row[target_name]) for row in valid_rows], dtype=float)

            for model_name, metadata in models.items():
                bundle = joblib.load(metadata["artifact_path"])
                model = bundle["model"]
                predictions = model.predict(matrix)
                metrics = regression_metrics(labels, predictions)
                metrics["mape"] = float(mean_absolute_percentage_error(labels, predictions))
                operational = benchmark_predict(lambda: model.predict(matrix))
                operational["throughput_rows_per_sec"] = round(
                    len(valid_rows) / max(operational["batch_latency_ms_mean"] / 1000.0, 1e-9),
                    3,
                )
                results.append(
                    {
                        "model_name": model_name,
                        "target": target_name,
                        **{key: round(value, 6) for key, value in metrics.items()},
                        **operational,
                    }
                )
                self._append_regression_errors(
                    rows=valid_rows,
                    model_name=model_name,
                    target_name=target_name,
                    labels=labels,
                    predictions=predictions,
                    error_rows=error_rows,
                )
                baseline_score = float(metrics["rmse"])
                for group_name, indices in groups.items():
                    ablated_matrix = matrix.copy()
                    ablated_matrix[:, indices] = 0.0
                    ablated_predictions = model.predict(ablated_matrix)
                    ablated_score = float(np.sqrt(mean_squared_error(labels, ablated_predictions)))
                    ablation_rows.append(
                        {
                            "model_name": model_name,
                            "target": target_name,
                            "group_name": group_name,
                            "baseline_metric": baseline_score,
                            "ablated_metric": ablated_score,
                            "delta": round(ablated_score - baseline_score, 6),
                        }
                    )

                svg_path = self._write_importance_chart(f"{model_name}_{target_name}", model, feature_columns)
                if svg_path:
                    feature_svg_paths.append(str(svg_path))

        return results

    def _evaluate_unsupervised_models(
        self,
        *,
        rows: list[dict[str, str]],
        matrix: np.ndarray,
        training_report: dict[str, Any],
    ) -> list[dict[str, Any]]:
        labels = np.array([int(row["is_anomaly"]) for row in rows], dtype=int)
        results: list[dict[str, Any]] = []
        for model_name, metadata in training_report.items():
            bundle = joblib.load(metadata["artifact_path"])
            model = bundle["model"]
            operational = benchmark_predict(lambda: model.predict(matrix))
            operational["throughput_rows_per_sec"] = round(
                len(rows) / max(operational["batch_latency_ms_mean"] / 1000.0, 1e-9),
                3,
            )
            if model_name == "isolation_forest":
                raw_predictions = model.predict(matrix)
                predictions = np.where(raw_predictions == -1, 1, 0)
                metrics = classification_metrics(labels, predictions)
                results.append(
                    {
                        "model_name": model_name,
                        **{key: (round(value, 6) if isinstance(value, float) else value) for key, value in metrics.items()},
                        **operational,
                    }
                )
            elif model_name == "kmeans":
                cluster_labels = model.predict(matrix)
                cluster_sizes = defaultdict(int)
                anomaly_counts = defaultdict(int)
                for index, cluster in enumerate(cluster_labels):
                    cluster_str = str(int(cluster))
                    cluster_sizes[cluster_str] += 1
                    anomaly_counts[cluster_str] += int(labels[index])
                results.append(
                    {
                        "model_name": model_name,
                        "accuracy": "",
                        "precision": "",
                        "recall": "",
                        "f1": "",
                        "roc_auc": "",
                        "cluster_summary": {
                            cluster: {
                                "size": cluster_sizes[cluster],
                                "anomaly_rate": round(anomaly_counts[cluster] / cluster_sizes[cluster], 6),
                            }
                            for cluster in sorted(cluster_sizes)
                        },
                        **operational,
                    }
                )
        return results

    def _append_threshold_sweeps(
        self,
        *,
        model_name: str,
        labels: np.ndarray,
        probabilities: np.ndarray | None,
        threshold_rows: list[dict[str, object]],
    ) -> None:
        if probabilities is None:
            return
        for step in range(1, 10):
            threshold = step / 10.0
            predictions = (probabilities >= threshold).astype(int)
            metrics = classification_metrics(labels, predictions, probabilities=probabilities)
            threshold_rows.append(
                {
                    "model_name": model_name,
                    "threshold": threshold,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics["roc_auc"],
                }
            )

    def _append_classification_errors(
        self,
        *,
        rows: list[dict[str, str]],
        model_name: str,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray | None,
        error_rows: list[dict[str, object]],
    ) -> None:
        for index, row in enumerate(rows):
            predicted = int(predictions[index])
            true_label = int(labels[index])
            error_rows.append(
                {
                    "model_name": model_name,
                    "window_start": row["window_start"],
                    "service_name": row["service_name"],
                    "host_id": row["host_id"],
                    "true_label": true_label,
                    "predicted_label": predicted,
                    "predicted_probability": "" if probabilities is None else round(float(probabilities[index]), 6),
                    "correct": int(predicted == true_label),
                }
            )

    def _append_regression_errors(
        self,
        *,
        rows: list[dict[str, str]],
        model_name: str,
        target_name: str,
        labels: np.ndarray,
        predictions: np.ndarray,
        error_rows: list[dict[str, object]],
    ) -> None:
        for index, row in enumerate(rows):
            actual = float(labels[index])
            predicted = float(predictions[index])
            error_rows.append(
                {
                    "model_name": model_name,
                    "target": target_name,
                    "window_start": row["window_start"],
                    "service_name": row["service_name"],
                    "host_id": row["host_id"],
                    "actual": round(actual, 6),
                    "predicted": round(predicted, 6),
                    "residual": round(predicted - actual, 6),
                    "absolute_error": round(abs(predicted - actual), 6),
                }
            )

    def _write_importance_chart(self, model_name: str, model: Any, feature_columns: list[str]) -> Path | None:
        importances = self._feature_importances(model, feature_columns)
        if not importances:
            return None
        svg = render_svg_bar_chart(
            f"Top Features: {model_name}",
            importances[:10],
        )
        path = self.importance_dir / f"{model_name}.svg"
        path.write_text(svg, encoding="utf-8")
        return path

    @staticmethod
    def _feature_importances(model: Any, feature_columns: list[str]) -> list[tuple[str, float]]:
        if hasattr(model, "feature_importances_"):
            values = [float(item) for item in model.feature_importances_]
        elif hasattr(model, "coef_"):
            raw = np.asarray(model.coef_)
            values = [float(abs(item)) for item in raw.reshape(-1)]
        else:
            return []
        pairs = list(zip(feature_columns, values))
        return sorted(pairs, key=lambda item: abs(item[1]), reverse=True)

    @staticmethod
    def _service_accuracy_breakdown(
        service_names: list[str],
        labels: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        per_service_totals = defaultdict(int)
        per_service_correct = defaultdict(int)
        for index, service_name in enumerate(service_names):
            per_service_totals[service_name] += 1
            per_service_correct[service_name] += int(int(labels[index]) == int(predictions[index]))
        return {
            service_name: round(per_service_correct[service_name] / per_service_totals[service_name], 6)
            for service_name in sorted(per_service_totals)
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

