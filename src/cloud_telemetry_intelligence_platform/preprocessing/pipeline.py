"""Cleaning and feature engineering pipeline for telemetry model training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
from typing import Any

from .transforms import (
    DEFAULT_METRIC_VALUES,
    canonicalize_metric,
    floor_timestamp,
    iter_window_range,
    rolling_stats,
    zscore,
)


@dataclass(slots=True)
class PreprocessingSummary:
    cleaned_path: str
    feature_path: str
    report_path: str
    rows_read: int
    cleaned_rows_written: int
    window_rows_written: int
    dropped_rows: int
    imputed_values: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


class PreprocessingPipeline:
    """Transform curated telemetry into model-ready window features."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.processed_root = data_root / "data" / "processed"
        self.curated_path = self.processed_root / "curated" / "telemetry_records.csv"
        self.cleaned_path = self.processed_root / "cleaned" / "telemetry_cleaned.csv"
        self.feature_path = self.processed_root / "features" / "window_features.csv"
        self.report_path = self.processed_root / "reports" / "preprocessing_report.json"

    def preprocess(
        self,
        *,
        curated_path: Path | None = None,
        window_minutes: int = 1,
        rolling_window_size: int = 3,
    ) -> PreprocessingSummary:
        self._ensure_layout()
        source_path = curated_path or self.curated_path
        rows = self._load_rows(source_path)
        cleaned_rows, dropped_rows = self._clean_rows(rows, window_minutes=window_minutes)
        metric_names = sorted({row["canonical_metric_name"] for row in cleaned_rows})
        feature_rows, imputed_values = self._build_feature_rows(
            cleaned_rows,
            metric_names=metric_names,
            window_minutes=window_minutes,
            rolling_window_size=rolling_window_size,
        )
        self._apply_normalization(feature_rows, metric_names=metric_names)
        self._write_csv(self.cleaned_path, cleaned_rows)
        self._write_csv(self.feature_path, feature_rows)
        self._write_report(
            rows_read=len(rows),
            cleaned_rows_written=len(cleaned_rows),
            window_rows_written=len(feature_rows),
            dropped_rows=dropped_rows,
            imputed_values=imputed_values,
            metric_names=metric_names,
            window_minutes=window_minutes,
            rolling_window_size=rolling_window_size,
        )
        return PreprocessingSummary(
            cleaned_path=str(self.cleaned_path),
            feature_path=str(self.feature_path),
            report_path=str(self.report_path),
            rows_read=len(rows),
            cleaned_rows_written=len(cleaned_rows),
            window_rows_written=len(feature_rows),
            dropped_rows=dropped_rows,
            imputed_values=imputed_values,
        )

    def _ensure_layout(self) -> None:
        for directory in (
            self.cleaned_path.parent,
            self.feature_path.parent,
            self.report_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _load_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _clean_rows(
        self,
        rows: list[dict[str, str]],
        *,
        window_minutes: int,
    ) -> tuple[list[dict[str, Any]], int]:
        cleaned_rows: list[dict[str, Any]] = []
        dropped_rows = 0
        for row in rows:
            timestamp = row.get("timestamp", "").strip()
            service_name = row.get("service_name", "").strip()
            host_id = row.get("host_id", "").strip()
            metric_name = row.get("metric_name", "").strip()

            if not timestamp or not service_name or not host_id or not metric_name:
                dropped_rows += 1
                continue

            try:
                parsed_timestamp = self._parse_timestamp(timestamp)
            except ValueError:
                dropped_rows += 1
                continue

            metric_value_raw = row.get("metric_value", "").strip()
            metric_value = float(metric_value_raw) if metric_value_raw else None
            canonical_name, canonical_unit, standardized_value = canonicalize_metric(
                metric_name,
                row.get("unit", ""),
                metric_value,
            )
            window_start = floor_timestamp(parsed_timestamp, window_minutes=window_minutes)

            cleaned_rows.append(
                {
                    "timestamp": parsed_timestamp.isoformat(),
                    "window_start": window_start.isoformat(),
                    "service_name": service_name,
                    "host_id": host_id,
                    "metric_name": metric_name,
                    "canonical_metric_name": canonical_name,
                    "standardized_metric_value": standardized_value if standardized_value is not None else "",
                    "standardized_unit": canonical_unit,
                    "event_type": row.get("event_type", "").strip() or "metric",
                    "event_summary": row.get("event_summary", "").strip(),
                    "anomaly_label": row.get("anomaly_label", "unknown").strip().lower() or "unknown",
                    "source_type": row.get("source_type", "").strip() or row.get("event_type", "").strip() or "metric",
                    "source_file": row.get("source_file", "").strip(),
                    "record_fingerprint": row.get("record_fingerprint", "").strip(),
                }
            )

        cleaned_rows.sort(key=lambda item: (item["service_name"], item["host_id"], item["timestamp"], item["canonical_metric_name"]))
        return cleaned_rows, dropped_rows

    def _build_feature_rows(
        self,
        cleaned_rows: list[dict[str, Any]],
        *,
        metric_names: list[str],
        window_minutes: int,
        rolling_window_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in cleaned_rows:
            grouped_rows[(row["service_name"], row["host_id"])].append(row)

        feature_rows: list[dict[str, Any]] = []
        imputed_values = 0

        for (service_name, host_id), group_rows in grouped_rows.items():
            windows: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
            for row in group_rows:
                window_start = self._parse_timestamp(row["window_start"])
                windows[window_start].append(row)

            ordered_window_starts = iter_window_range(
                min(windows.keys()),
                max(windows.keys()),
                window_minutes=window_minutes,
            )
            prior_values = dict(DEFAULT_METRIC_VALUES)
            metric_history: dict[str, list[float]] = defaultdict(list)
            burst_count = 0
            group_feature_rows: list[dict[str, Any]] = []

            for window_start in ordered_window_starts:
                window_rows = windows.get(window_start, [])
                row: dict[str, Any] = {
                    "window_start": window_start.isoformat(),
                    "window_minutes": window_minutes,
                    "service_name": service_name,
                    "host_id": host_id,
                    "event_count": sum(1 for item in window_rows if item["event_type"] == "event"),
                    "event_frequency": 0.0,
                    "event_summaries": " | ".join(sorted({item["event_summary"] for item in window_rows if item["event_summary"]})),
                    "is_anomaly": 1 if any(item["anomaly_label"] == "anomaly" for item in window_rows) else 0,
                    "target_next_latency_ms": "",
                    "target_next_throughput_rps": "",
                }

                per_metric_values: dict[str, list[float]] = defaultdict(list)
                for item in window_rows:
                    value = item["standardized_metric_value"]
                    if value == "":
                        continue
                    per_metric_values[item["canonical_metric_name"]].append(float(value))

                for metric_name in metric_names:
                    metric_values = per_metric_values.get(metric_name, [])
                    if metric_values:
                        value = sum(metric_values) / len(metric_values)
                        was_imputed = 0
                    else:
                        value = prior_values.get(metric_name, DEFAULT_METRIC_VALUES.get(metric_name, 0.0))
                        was_imputed = 1
                        imputed_values += 1

                    prior_values[metric_name] = value
                    metric_history[metric_name].append(value)
                    recent_values = metric_history[metric_name][-rolling_window_size:]
                    stats = rolling_stats(recent_values)

                    row[metric_name] = round(value, 6)
                    row[f"{metric_name}_was_imputed"] = was_imputed
                    row[f"{metric_name}_roll_mean"] = round(stats["mean"], 6)
                    row[f"{metric_name}_roll_std"] = round(stats["std"], 6)
                    row[f"{metric_name}_roll_min"] = round(stats["min"], 6)
                    row[f"{metric_name}_roll_max"] = round(stats["max"], 6)
                    row[f"{metric_name}_roll_slope"] = round(stats["slope"], 6)

                throughput = row.get("throughput_rps", 0.0) or 0.0
                row["request_error_ratio"] = round(
                    row.get("error_rate", 0.0) or (
                        (row.get("request_failures", 0.0) / throughput) if throughput else 0.0
                    ),
                    6,
                )
                row["cpu_throughput_imbalance"] = round(
                    (row.get("cpu_utilization", 0.0) / throughput) if throughput else 0.0,
                    9,
                )
                row["short_term_latency_drift"] = round(
                    row.get("latency_ms", 0.0) - row.get("latency_ms_roll_mean", 0.0),
                    6,
                )
                if row.get("packet_drop_ratio", 0.0) > 0.01:
                    burst_count += 1
                else:
                    burst_count = 0
                row["packet_drop_burst_count"] = burst_count
                row["event_frequency"] = round(row["event_count"] / window_minutes, 6)
                group_feature_rows.append(row)

            for index, row in enumerate(group_feature_rows):
                if index + 1 < len(group_feature_rows):
                    next_row = group_feature_rows[index + 1]
                    row["target_next_latency_ms"] = next_row.get("latency_ms", "")
                    row["target_next_throughput_rps"] = next_row.get("throughput_rps", "")

            feature_rows.extend(group_feature_rows)

        feature_rows.sort(key=lambda item: (item["service_name"], item["host_id"], item["window_start"]))
        return feature_rows, imputed_values

    def _apply_normalization(self, feature_rows: list[dict[str, Any]], *, metric_names: list[str]) -> None:
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in feature_rows]
            for row in feature_rows:
                row[f"{metric_name}_zscore"] = round(zscore(float(row[metric_name]), values), 6)

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_report(
        self,
        *,
        rows_read: int,
        cleaned_rows_written: int,
        window_rows_written: int,
        dropped_rows: int,
        imputed_values: int,
        metric_names: list[str],
        window_minutes: int,
        rolling_window_size: int,
    ) -> None:
        payload = {
            "rows_read": rows_read,
            "cleaned_rows_written": cleaned_rows_written,
            "window_rows_written": window_rows_written,
            "dropped_rows": dropped_rows,
            "imputed_values": imputed_values,
            "metric_names": metric_names,
            "window_minutes": window_minutes,
            "rolling_window_size": rolling_window_size,
        }
        self.report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        return datetime.fromisoformat(candidate)

