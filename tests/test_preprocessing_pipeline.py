from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud_telemetry_intelligence_platform.preprocessing.pipeline import PreprocessingPipeline


CURATED_FIELDS = [
    "timestamp",
    "service_name",
    "host_id",
    "metric_name",
    "metric_value",
    "unit",
    "event_type",
    "event_summary",
    "anomaly_label",
    "source_type",
    "source_file",
    "record_fingerprint",
]


class PreprocessingPipelineTests(unittest.TestCase):
    def test_builds_cleaned_and_windowed_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            curated_dir = root / "data" / "processed" / "curated"
            curated_dir.mkdir(parents=True, exist_ok=True)
            curated_path = curated_dir / "telemetry_records.csv"

            rows = [
                ["2026-04-14T10:00:05Z", "edge-api", "host-01", "cpu_pct", "60", "percent", "metric", "", "normal", "metric", "sample.csv", "a1"],
                ["2026-04-14T10:00:10Z", "edge-api", "host-01", "throughput_rps", "1000", "rps", "metric", "", "normal", "metric", "sample.csv", "a2"],
                ["2026-04-14T10:00:20Z", "edge-api", "host-01", "latency_ms", "100", "ms", "metric", "", "normal", "metric", "sample.csv", "a3"],
                ["2026-04-14T10:01:05Z", "edge-api", "host-01", "cpu_pct", "80", "percent", "metric", "", "normal", "metric", "sample.csv", "b1"],
                ["2026-04-14T10:01:10Z", "edge-api", "host-01", "throughput_rps", "", "rps", "metric", "", "normal", "metric", "sample.csv", "b2"],
                ["2026-04-14T10:01:15Z", "edge-api", "host-01", "latency_ms", "150", "ms", "metric", "", "normal", "metric", "sample.csv", "b3"],
                ["2026-04-14T10:01:25Z", "edge-api", "host-01", "packet_drop_pct", "2", "percent", "event", "packet loss burst", "anomaly", "event", "sample.csv", "b4"],
                ["2026-04-14T10:02:05Z", "edge-api", "host-01", "cpu_pct", "50", "percent", "metric", "", "normal", "metric", "sample.csv", "c1"],
                ["2026-04-14T10:02:10Z", "edge-api", "host-01", "throughput_rps", "800", "rps", "metric", "", "normal", "metric", "sample.csv", "c2"],
                ["2026-04-14T10:02:15Z", "edge-api", "host-01", "latency_ms", "90", "ms", "metric", "", "normal", "metric", "sample.csv", "c3"],
            ]

            with curated_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(CURATED_FIELDS)
                writer.writerows(rows)

            pipeline = PreprocessingPipeline(root)
            summary = pipeline.preprocess()

            self.assertEqual(summary.rows_read, 10)
            self.assertEqual(summary.cleaned_rows_written, 10)
            self.assertEqual(summary.window_rows_written, 3)
            self.assertGreaterEqual(summary.imputed_values, 1)

            cleaned_path = root / "data" / "processed" / "cleaned" / "telemetry_cleaned.csv"
            features_path = root / "data" / "processed" / "features" / "window_features.csv"
            report_path = root / "data" / "processed" / "reports" / "preprocessing_report.json"

            with cleaned_path.open("r", encoding="utf-8", newline="") as handle:
                cleaned_rows = list(csv.DictReader(handle))
            cpu_row = next(row for row in cleaned_rows if row["metric_name"] == "cpu_pct")
            self.assertEqual(cpu_row["canonical_metric_name"], "cpu_utilization")
            self.assertEqual(cpu_row["standardized_unit"], "ratio")
            self.assertEqual(cpu_row["standardized_metric_value"], "0.6")

            with features_path.open("r", encoding="utf-8", newline="") as handle:
                feature_rows = list(csv.DictReader(handle))
            first_row, second_row, third_row = feature_rows
            self.assertEqual(first_row["target_next_latency_ms"], "150.0")
            self.assertEqual(second_row["is_anomaly"], "1")
            self.assertEqual(second_row["throughput_rps"], "1000.0")
            self.assertEqual(second_row["throughput_rps_was_imputed"], "1")
            self.assertEqual(second_row["packet_drop_burst_count"], "1")
            self.assertEqual(second_row["event_summaries"], "packet loss burst")
            self.assertEqual(second_row["latency_ms_roll_mean"], "125.0")
            self.assertIn("cpu_utilization_zscore", second_row)
            self.assertEqual(third_row["target_next_latency_ms"], "")

            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report_payload["window_rows_written"], 3)

    def test_drops_rows_with_missing_critical_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            curated_dir = root / "data" / "processed" / "curated"
            curated_dir.mkdir(parents=True, exist_ok=True)
            curated_path = curated_dir / "telemetry_records.csv"

            with curated_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(CURATED_FIELDS)
                writer.writerow(["", "edge-api", "host-01", "cpu_pct", "60", "percent", "metric", "", "normal", "metric", "sample.csv", "a1"])
                writer.writerow(["2026-04-14T10:00:05Z", "edge-api", "host-01", "cpu_pct", "60", "percent", "metric", "", "normal", "metric", "sample.csv", "a2"])

            pipeline = PreprocessingPipeline(root)
            summary = pipeline.preprocess()
            self.assertEqual(summary.rows_read, 2)
            self.assertEqual(summary.cleaned_rows_written, 1)
            self.assertEqual(summary.dropped_rows, 1)


if __name__ == "__main__":
    unittest.main()
