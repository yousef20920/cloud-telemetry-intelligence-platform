"""Source loaders for telemetry ingestion."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
from typing import Iterator


def load_source_records(path: Path, *, sql_table: str = "telemetry") -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return list(_load_csv_records(path))
    if suffix in {".jsonl", ".ndjson"}:
        return list(_load_jsonl_records(path))
    if suffix == ".json":
        return list(_load_json_records(path))
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        return list(_load_sqlite_records(path, sql_table=sql_table))
    raise ValueError(f"Unsupported input type for {path.name}")


def _load_csv_records(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def _load_jsonl_records(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield json.loads(text)


def _load_json_records(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for item in payload:
            yield dict(item)
        return
    raise ValueError(f"Expected a JSON array in {path.name}")


def _load_sqlite_records(path: Path, *, sql_table: str) -> Iterator[dict[str, object]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.execute(f"SELECT * FROM {sql_table}")
        for row in cursor.fetchall():
            yield dict(row)
    finally:
        connection.close()

