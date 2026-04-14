"""Validation helpers for telemetry ingestion."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import re
from typing import Iterable

from .models import TelemetryRecord, ValidationIssue

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")
NULL_SPIKE_THRESHOLD = 0.2
ALLOWED_SOURCE_TYPES = {"metric", "event"}
ALLOWED_ANOMALY_LABELS = {"normal", "anomaly", "unknown"}
METRIC_RANGES = {
    "cpu_pct": (0.0, 100.0),
    "memory_pct": (0.0, 100.0),
    "latency_ms": (0.0, 600000.0),
    "packet_drop_pct": (0.0, 100.0),
    "error_rate": (0.0, 1.0),
    "throughput_rps": (0.0, 10000000.0),
    "request_failures": (0.0, 10000000.0),
}


def normalize_timestamp(raw_timestamp: str) -> str:
    candidate = raw_timestamp.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    parsed = datetime.fromisoformat(candidate)
    return parsed.isoformat()


def build_fingerprint(
    timestamp: str,
    service_name: str,
    host_id: str,
    metric_name: str,
    metric_value: float,
    event_type: str,
) -> str:
    payload = "|".join(
        [
            timestamp,
            service_name,
            host_id,
            metric_name,
            f"{metric_value:.10f}",
            event_type,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_record(
    raw: dict[str, object],
    *,
    row_number: int,
    source_file: str,
) -> tuple[TelemetryRecord | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []

    def add_issue(code: str, message: str) -> None:
        issues.append(
            ValidationIssue(
                source_file=source_file,
                row_number=row_number,
                code=code,
                message=message,
            )
        )

    def get_string(field: str, *, required: bool = True, default: str = "") -> str:
        value = raw.get(field, default)
        if value is None:
            value = default
        text = str(value).strip()
        if required and not text:
            add_issue("missing_field", f"Missing required field '{field}'")
        return text

    timestamp_raw = get_string("timestamp")
    service_name = get_string("service_name")
    host_id = get_string("host_id")
    metric_name = get_string("metric_name")
    unit = get_string("unit", required=False, default="")
    event_type = get_string("event_type", required=False, default="metric")
    event_summary = get_string("event_summary", required=False, default="")
    anomaly_label = get_string("anomaly_label", required=False, default="unknown").lower()
    metric_value_raw = raw.get("metric_value", "")

    try:
        timestamp = normalize_timestamp(timestamp_raw)
    except Exception:
        timestamp = ""
        add_issue("invalid_timestamp", "Timestamp must be ISO-8601 parseable")

    if service_name and not IDENTIFIER_RE.fullmatch(service_name):
        add_issue("invalid_service_name", "service_name contains unsupported characters")
    if host_id and not IDENTIFIER_RE.fullmatch(host_id):
        add_issue("invalid_host_id", "host_id contains unsupported characters")

    try:
        metric_value = float(metric_value_raw)
    except (TypeError, ValueError):
        metric_value = 0.0
        add_issue("invalid_metric_value", "metric_value must be numeric")

    if metric_name in METRIC_RANGES:
        lower, upper = METRIC_RANGES[metric_name]
        if not (lower <= metric_value <= upper):
            add_issue(
                "invalid_metric_range",
                f"{metric_name} must be between {lower} and {upper}",
            )

    if event_type not in ALLOWED_SOURCE_TYPES:
        add_issue("invalid_event_type", "event_type must be 'metric' or 'event'")

    if anomaly_label not in ALLOWED_ANOMALY_LABELS:
        add_issue("invalid_anomaly_label", "anomaly_label must be normal, anomaly, or unknown")

    if issues:
        return None, issues

    return (
        TelemetryRecord(
            timestamp=timestamp,
            service_name=service_name,
            host_id=host_id,
            metric_name=metric_name,
            metric_value=metric_value,
            unit=unit,
            event_type=event_type,
            event_summary=event_summary,
            anomaly_label=anomaly_label,
            source_type=event_type,
            source_file=source_file,
            record_fingerprint=build_fingerprint(
                timestamp=timestamp,
                service_name=service_name,
                host_id=host_id,
                metric_name=metric_name,
                metric_value=metric_value,
                event_type=event_type,
            ),
        ),
        issues,
    )


def summarize_null_spikes(
    rows: Iterable[dict[str, object]],
    *,
    required_fields: tuple[str, ...] = ("timestamp", "service_name", "host_id", "metric_name", "metric_value"),
) -> dict[str, float]:
    rows_list = list(rows)
    if not rows_list:
        return {}

    counter: Counter[str] = Counter()
    for row in rows_list:
        for field in required_fields:
            value = row.get(field)
            if value is None or str(value).strip() == "":
                counter[field] += 1

    total = float(len(rows_list))
    ratios = {field: counter[field] / total for field in required_fields}
    return {field: ratio for field, ratio in ratios.items() if ratio >= NULL_SPIKE_THRESHOLD}

