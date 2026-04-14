"""Reusable preprocessing transforms for telemetry windows."""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import fmean, pstdev


METRIC_SPECS = {
    "cpu_pct": {"canonical_name": "cpu_utilization", "canonical_unit": "ratio"},
    "memory_pct": {"canonical_name": "memory_utilization", "canonical_unit": "ratio"},
    "packet_drop_pct": {"canonical_name": "packet_drop_ratio", "canonical_unit": "ratio"},
    "error_rate": {"canonical_name": "error_rate", "canonical_unit": "ratio"},
    "latency_ms": {"canonical_name": "latency_ms", "canonical_unit": "ms"},
    "throughput_rps": {"canonical_name": "throughput_rps", "canonical_unit": "rps"},
    "request_failures": {"canonical_name": "request_failures", "canonical_unit": "count"},
}

DEFAULT_METRIC_VALUES = {
    "cpu_utilization": 0.0,
    "memory_utilization": 0.0,
    "packet_drop_ratio": 0.0,
    "error_rate": 0.0,
    "latency_ms": 0.0,
    "throughput_rps": 0.0,
    "request_failures": 0.0,
}


def canonicalize_metric(metric_name: str, unit: str, value: float | None) -> tuple[str, str, float | None]:
    metric_name = metric_name.strip()
    spec = METRIC_SPECS.get(metric_name)
    canonical_name = spec["canonical_name"] if spec else metric_name
    canonical_unit = spec["canonical_unit"] if spec else unit.strip().lower()
    if value is None:
        return canonical_name, canonical_unit, None

    normalized_unit = unit.strip().lower()
    normalized_value = value

    if canonical_unit == "ratio":
        if normalized_unit in {"percent", "pct", "%"}:
            normalized_value = value / 100.0
        elif normalized_unit in {"ratio", ""}:
            normalized_value = value
    elif canonical_unit == "ms":
        if normalized_unit in {"s", "sec", "second", "seconds"}:
            normalized_value = value * 1000.0
        elif normalized_unit in {"us", "microsecond", "microseconds"}:
            normalized_value = value / 1000.0
    elif canonical_unit == "rps":
        if normalized_unit in {"per_min", "rpm"}:
            normalized_value = value / 60.0

    return canonical_name, canonical_unit, normalized_value


def floor_timestamp(timestamp: datetime, *, window_minutes: int) -> datetime:
    floored = timestamp.replace(second=0, microsecond=0)
    minute = floored.minute - (floored.minute % window_minutes)
    return floored.replace(minute=minute)


def iter_window_range(start: datetime, end: datetime, *, window_minutes: int) -> list[datetime]:
    windows: list[datetime] = []
    current = start
    step = timedelta(minutes=window_minutes)
    while current <= end:
        windows.append(current)
        current += step
    return windows


def rolling_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "slope": 0.0,
        }

    if len(values) == 1:
        return {
            "mean": values[0],
            "std": 0.0,
            "min": values[0],
            "max": values[0],
            "slope": 0.0,
        }

    return {
        "mean": fmean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
        "slope": (values[-1] - values[0]) / (len(values) - 1),
    }


def zscore(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = fmean(values)
    std_value = pstdev(values) if len(values) > 1 else 0.0
    if std_value == 0.0:
        return 0.0
    return (value - mean_value) / std_value

