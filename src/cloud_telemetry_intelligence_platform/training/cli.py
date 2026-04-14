"""CLI entrypoint for telemetry model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import TrainingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telemetry model training")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root where model artifacts and reports will be written",
    )
    parser.add_argument(
        "--feature-path",
        default="",
        help="Optional features CSV path. Defaults to data/processed/features/window_features.csv",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.4,
        help="Holdout ratio for classifier and regressor evaluation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic training runs",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = TrainingPipeline(Path(args.project_root).resolve())
    feature_path = Path(args.feature_path).resolve() if args.feature_path else None
    summary = pipeline.train(
        feature_path=feature_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(summary.to_row(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

