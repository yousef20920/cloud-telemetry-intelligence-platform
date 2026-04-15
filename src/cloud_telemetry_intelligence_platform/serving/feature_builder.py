"""Feature construction for online inference windows."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from ..preprocessing.transforms import DEFAULT_METRIC_VALUES, canonicalize_metric, floor_timestamp, rolling_stats, zscore
from .schemas import HistoryWindow, InferenceWindow


METRIC_ORDER = tuple(DEFAULT_METRIC_VALUES.keys())


def build_feature_map(window: InferenceWindow, *, feature_columns: list[str]) -> dict[str, float]:
    metric_history = _history_by_metric(window.history)
    per_metric_values: dict[str, list[float]] = defaultdict(list)
    event_count = 0
    event_summaries: set[str] = set()

    for observation in window.observations:
        canonical_name, _, normalized_value = canonicalize_metric(
            observation.metric_name,
            observation.unit,
            observation.metric_value,
        )
        if normalized_value is not None:
            per_metric_values[canonical_name].append(float(normalized_value))
        if observation.event_type == "event":
            event_count += 1
        if observation.event_summary:
            event_summaries.add(observation.event_summary)

    feature_map: dict[str, float] = {column: 0.0 for column in feature_columns}
    for metric_name in METRIC_ORDER:
        current_values = per_metric_values.get(metric_name, [])
        history_values = metric_history.get(metric_name, [])
        if current_values:
            current_value = float(sum(current_values) / len(current_values))
            was_imputed = 0.0
        elif history_values:
            current_value = float(history_values[-1])
            was_imputed = 1.0
        else:
            current_value = float(DEFAULT_METRIC_VALUES[metric_name])
            was_imputed = 1.0

        rolling_values = (history_values + [current_value])[-window.rolling_window_size :]
        stats = rolling_stats(rolling_values)
        zscore_values = history_values + [current_value]

        _set_if_present(feature_map, metric_name, current_value)
        _set_if_present(feature_map, f"{metric_name}_was_imputed", was_imputed)
        _set_if_present(feature_map, f"{metric_name}_roll_mean", float(stats["mean"]))
        _set_if_present(feature_map, f"{metric_name}_roll_std", float(stats["std"]))
        _set_if_present(feature_map, f"{metric_name}_roll_min", float(stats["min"]))
        _set_if_present(feature_map, f"{metric_name}_roll_max", float(stats["max"]))
        _set_if_present(feature_map, f"{metric_name}_roll_slope", float(stats["slope"]))
        _set_if_present(feature_map, f"{metric_name}_zscore", float(zscore(current_value, zscore_values)))

    throughput = feature_map.get("throughput_rps", 0.0)
    request_failures = feature_map.get("request_failures", 0.0)
    error_rate = feature_map.get("error_rate", 0.0)
    packet_drop_ratio = feature_map.get("packet_drop_ratio", 0.0)
    latency_ms = feature_map.get("latency_ms", 0.0)
    latency_roll_mean = feature_map.get("latency_ms_roll_mean", 0.0)

    _set_if_present(feature_map, "event_count", float(event_count))
    _set_if_present(feature_map, "event_frequency", float(event_count / window.window_minutes))
    _set_if_present(
        feature_map,
        "request_error_ratio",
        float(error_rate or ((request_failures / throughput) if throughput else 0.0)),
    )
    _set_if_present(
        feature_map,
        "cpu_throughput_imbalance",
        float((feature_map.get("cpu_utilization", 0.0) / throughput) if throughput else 0.0),
    )
    _set_if_present(feature_map, "short_term_latency_drift", float(latency_ms - latency_roll_mean))
    _set_if_present(feature_map, "packet_drop_burst_count", float(1 if packet_drop_ratio > 0.01 else 0))

    return feature_map


def feature_vector(feature_map: dict[str, float], *, feature_columns: list[str]) -> np.ndarray:
    return np.array([[float(feature_map.get(column, 0.0)) for column in feature_columns]], dtype=float)


def preview_feature_map(feature_map: dict[str, float]) -> dict[str, float | int]:
    preview_keys = [
        "cpu_utilization",
        "memory_utilization",
        "latency_ms",
        "throughput_rps",
        "packet_drop_ratio",
        "error_rate",
        "request_failures",
        "request_error_ratio",
        "event_count",
        "packet_drop_burst_count",
    ]
    preview: dict[str, float | int] = {}
    for key in preview_keys:
        if key in feature_map:
            value = feature_map[key]
            preview[key] = int(value) if float(value).is_integer() else round(float(value), 6)
    return preview


def normalize_window_start(raw_window_start: str, *, window_minutes: int) -> str:
    candidate = raw_window_start.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    parsed = datetime.fromisoformat(candidate)
    return floor_timestamp(parsed, window_minutes=window_minutes).isoformat()


def _history_by_metric(history: list[HistoryWindow]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for item in history:
        for metric_name in METRIC_ORDER:
            value = getattr(item, metric_name)
            if value is not None:
                values[metric_name].append(float(value))
    return values


def _set_if_present(feature_map: dict[str, float], key: str, value: float) -> None:
    if key in feature_map:
        feature_map[key] = float(value)

