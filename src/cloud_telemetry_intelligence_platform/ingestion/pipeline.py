"""Batch ingestion pipeline for telemetry sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil
from pathlib import Path

from .models import TelemetryRecord
from .sources import load_source_records
from .validation import summarize_null_spikes, validate_record


@dataclass(slots=True)
class SourceIngestionSummary:
    source_file: str
    checksum: str
    archived_raw_path: str
    status: str
    total_rows_seen: int
    rows_written: int
    duplicate_rows: int
    validation_errors: int
    null_spikes: dict[str, float]
    report_path: str

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class IngestionSummary:
    run_id: str
    curated_path: str
    sources: list[SourceIngestionSummary]

    def to_row(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "curated_path": self.curated_path,
            "sources": [source.to_row() for source in self.sources],
        }


class IngestionPipeline:
    """Ingest telemetry files into archived raw and curated processed outputs."""

    curated_filename = "telemetry_records.csv"

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.raw_root = data_root / "data" / "raw"
        self.processed_root = data_root / "data" / "processed"
        self.curated_root = self.processed_root / "curated"
        self.report_root = self.processed_root / "reports"
        self.manifest_path = self.processed_root / "manifests" / "ingestion_manifest.json"

    def ingest(self, inputs: list[Path], *, sql_table: str = "telemetry") -> IngestionSummary:
        if not inputs:
            raise ValueError("At least one input source is required")

        self._ensure_layout()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest = self._load_manifest()
        existing_fingerprints = self._load_existing_fingerprints()
        source_summaries: list[SourceIngestionSummary] = []

        for input_path in inputs:
            source_path = input_path.resolve()
            checksum = self._checksum_file(source_path)
            archived_path = self._archive_raw_file(source_path, run_id=run_id, checksum=checksum)
            report_path = self.report_root / f"{source_path.stem}_{checksum[:10]}.json"

            if checksum in manifest["processed_checksums"]:
                source_summary = SourceIngestionSummary(
                    source_file=str(source_path),
                    checksum=checksum,
                    archived_raw_path=str(archived_path),
                    status="skipped_existing_checksum",
                    total_rows_seen=0,
                    rows_written=0,
                    duplicate_rows=0,
                    validation_errors=0,
                    null_spikes={},
                    report_path=str(report_path),
                )
                self._write_report(report_path, source_summary.to_row())
                source_summaries.append(source_summary)
                continue

            raw_rows = load_source_records(source_path, sql_table=sql_table)
            null_spikes = summarize_null_spikes(raw_rows)

            accepted_records: list[TelemetryRecord] = []
            validation_errors: list[dict[str, object]] = []
            seen_in_file: set[str] = set()
            duplicate_rows = 0

            for row_number, raw_row in enumerate(raw_rows, start=2):
                record, issues = validate_record(
                    raw_row,
                    row_number=row_number,
                    source_file=source_path.name,
                )
                if issues:
                    validation_errors.extend(issue.to_row() for issue in issues)
                    continue

                assert record is not None
                if record.record_fingerprint in seen_in_file or record.record_fingerprint in existing_fingerprints:
                    duplicate_rows += 1
                    continue

                seen_in_file.add(record.record_fingerprint)
                existing_fingerprints.add(record.record_fingerprint)
                accepted_records.append(record)

            rows_written = self._append_curated_records(accepted_records)
            source_summary = SourceIngestionSummary(
                source_file=str(source_path),
                checksum=checksum,
                archived_raw_path=str(archived_path),
                status="ingested",
                total_rows_seen=len(raw_rows),
                rows_written=rows_written,
                duplicate_rows=duplicate_rows,
                validation_errors=len(validation_errors),
                null_spikes=null_spikes,
                report_path=str(report_path),
            )
            self._write_report(
                report_path,
                {
                    **source_summary.to_row(),
                    "validation_issues": validation_errors,
                },
            )

            manifest["processed_checksums"][checksum] = {
                "source_file": str(source_path),
                "archived_raw_path": str(archived_path),
                "report_path": str(report_path),
                "run_id": run_id,
            }
            source_summaries.append(source_summary)

        self._write_manifest(manifest)
        return IngestionSummary(
            run_id=run_id,
            curated_path=str(self.curated_root / self.curated_filename),
            sources=source_summaries,
        )

    def _ensure_layout(self) -> None:
        for directory in (
            self.raw_root,
            self.curated_root,
            self.report_root,
            self.manifest_path.parent,
            self.data_root / "data" / "synthetic",
            self.data_root / "notebooks",
            self.data_root / "configs",
            self.data_root / "scripts",
            self.data_root / "tests",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _load_manifest(self) -> dict[str, dict[str, object]]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"processed_checksums": {}}

    def _write_manifest(self, manifest: dict[str, object]) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_existing_fingerprints(self) -> set[str]:
        curated_path = self.curated_root / self.curated_filename
        if not curated_path.exists():
            return set()

        with curated_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return {row["record_fingerprint"] for row in reader if row.get("record_fingerprint")}

    def _append_curated_records(self, records: list[TelemetryRecord]) -> int:
        if not records:
            return 0

        curated_path = self.curated_root / self.curated_filename
        fieldnames = list(records[0].to_row().keys())
        write_header = not curated_path.exists()

        with curated_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for record in records:
                writer.writerow(record.to_row())
        return len(records)

    def _archive_raw_file(self, source_path: Path, *, run_id: str, checksum: str) -> Path:
        archive_dir = self.raw_root / run_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived_path = archive_dir / f"{checksum[:10]}_{source_path.name}"
        shutil.copy2(source_path, archived_path)
        return archived_path

    def _write_report(self, report_path: Path, payload: dict[str, object]) -> None:
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _checksum_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

