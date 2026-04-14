"""CLI entrypoint for telemetry preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import PreprocessingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telemetry preprocessing pipeline")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root where processed outputs will be written",
    )
    parser.add_argument(
        "--curated-path",
        default="",
        help="Optional curated telemetry CSV path. Defaults to data/processed/curated/telemetry_records.csv",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=1,
        help="Window size in minutes for resampling",
    )
    parser.add_argument(
        "--rolling-window-size",
        type=int,
        default=3,
        help="Window count used for rolling statistics",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = PreprocessingPipeline(Path(args.project_root).resolve())
    curated_path = Path(args.curated_path).resolve() if args.curated_path else None
    summary = pipeline.preprocess(
        curated_path=curated_path,
        window_minutes=args.window_minutes,
        rolling_window_size=args.rolling_window_size,
    )
    print(json.dumps(summary.to_row(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

