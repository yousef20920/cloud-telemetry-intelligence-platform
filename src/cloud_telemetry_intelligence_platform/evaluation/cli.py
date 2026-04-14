"""CLI entrypoint for telemetry model evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import EvaluationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telemetry model evaluation")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root where reports and dashboards will be written",
    )
    parser.add_argument(
        "--feature-path",
        default="",
        help="Optional features CSV path. Defaults to data/processed/features/window_features.csv",
    )
    parser.add_argument(
        "--training-report-path",
        default="",
        help="Optional training report JSON path. Defaults to data/processed/reports/training_report.json",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = EvaluationPipeline(Path(args.project_root).resolve())
    feature_path = Path(args.feature_path).resolve() if args.feature_path else None
    training_report_path = Path(args.training_report_path).resolve() if args.training_report_path else None
    summary = pipeline.evaluate(
        feature_path=feature_path,
        training_report_path=training_report_path,
    )
    print(json.dumps(summary.to_row(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

