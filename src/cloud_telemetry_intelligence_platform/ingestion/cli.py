"""CLI entrypoint for Phase 1 ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import IngestionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telemetry ingestion pipeline")
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        help="Path to a CSV, JSON, JSONL, or SQLite telemetry source",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root where data/raw and data/processed will be written",
    )
    parser.add_argument(
        "--sql-table",
        default="telemetry",
        help="SQLite table name when ingesting .db/.sqlite sources",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = IngestionPipeline(Path(args.project_root).resolve())
    summary = pipeline.ingest([Path(item) for item in args.inputs], sql_table=args.sql_table)
    print(json.dumps(summary.to_row(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

