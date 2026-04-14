"""Core data models for telemetry ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TelemetryRecord:
    """Normalized telemetry row shared by all source loaders."""

    timestamp: str
    service_name: str
    host_id: str
    metric_name: str
    metric_value: float
    unit: str
    event_type: str
    event_summary: str
    anomaly_label: str
    source_type: str
    source_file: str
    record_fingerprint: str

    def to_row(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(slots=True)
class ValidationIssue:
    """Structured validation issue for auditability."""

    source_file: str
    row_number: int
    code: str
    message: str

    def to_row(self) -> dict[str, str | int]:
        return {
            "source_file": self.source_file,
            "row_number": self.row_number,
            "code": self.code,
            "message": self.message,
        }

