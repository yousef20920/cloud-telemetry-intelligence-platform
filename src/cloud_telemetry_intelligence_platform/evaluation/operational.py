"""Operational analysis helpers for telemetry model evaluation."""

from __future__ import annotations

from statistics import fmean
import time
from typing import Callable


DERIVED_FEATURES = {
    "event_count",
    "event_frequency",
    "request_error_ratio",
    "cpu_throughput_imbalance",
    "short_term_latency_drift",
    "packet_drop_burst_count",
}


def benchmark_predict(callable_predict: Callable[[], object], *, repeats: int = 25) -> dict[str, float]:
    durations_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_predict()
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(durations_ms)
    p95_index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
    return {
        "batch_latency_ms_mean": round(fmean(durations_ms), 6),
        "batch_latency_ms_p95": round(ordered[p95_index], 6),
        "batch_latency_ms_min": round(ordered[0], 6),
        "batch_latency_ms_max": round(ordered[-1], 6),
    }


def feature_groups(feature_columns: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "base_metrics": [],
        "rolling_stats": [],
        "imputation_flags": [],
        "normalized_scores": [],
        "derived_operational": [],
    }
    for index, name in enumerate(feature_columns):
        if name.endswith("_was_imputed"):
            groups["imputation_flags"].append(index)
        elif name.endswith("_zscore"):
            groups["normalized_scores"].append(index)
        elif "_roll_" in name:
            groups["rolling_stats"].append(index)
        elif name in DERIVED_FEATURES:
            groups["derived_operational"].append(index)
        else:
            groups["base_metrics"].append(index)
    return {group: positions for group, positions in groups.items() if positions}

