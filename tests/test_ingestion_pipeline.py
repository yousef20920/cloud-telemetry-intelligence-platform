from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud_telemetry_intelligence_platform.ingestion.pipeline import IngestionPipeline


class IngestionPipelineTests(unittest.TestCase):
    def test_ingests_csv_and_jsonl_with_validation_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "metrics.csv"
            jsonl_path = root / "events.jsonl"

            csv_path.write_text(
                "\n".join(
                    [
                        "timestamp,service_name,host_id,metric_name,metric_value,unit,event_type,event_summary,anomaly_label",
                        "2026-04-14T10:00:00Z,edge-api,host-01,cpu_pct,71.4,percent,metric,,normal",
                        "2026-04-14T10:00:00Z,edge-api,host-01,cpu_pct,71.4,percent,metric,,normal",
                        "2026-04-14T10:02:00Z,bad service,host-01,latency_ms,42,ms,metric,,normal",
                    ]
                ),
                encoding="utf-8",
            )
            jsonl_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": "2026-04-14T10:03:00Z",
                                "service_name": "mesh-gateway",
                                "host_id": "gateway-03",
                                "metric_name": "throughput_rps",
                                "metric_value": 12450,
                                "unit": "rps",
                                "event_type": "metric",
                                "event_summary": "",
                                "anomaly_label": "normal",
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "bad-timestamp",
                                "service_name": "mesh-gateway",
                                "host_id": "gateway-03",
                                "metric_name": "latency_ms",
                                "metric_value": 91,
                                "unit": "ms",
                                "event_type": "metric",
                                "event_summary": "",
                                "anomaly_label": "normal",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            pipeline = IngestionPipeline(root)
            summary = pipeline.ingest([csv_path, jsonl_path])

            curated_path = root / "data" / "processed" / "curated" / "telemetry_records.csv"
            self.assertTrue(curated_path.exists())
            with curated_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["service_name"] for row in rows}, {"edge-api", "mesh-gateway"})

            first_source, second_source = summary.sources
            self.assertEqual(first_source.rows_written, 1)
            self.assertEqual(first_source.duplicate_rows, 1)
            self.assertEqual(first_source.validation_errors, 1)
            self.assertEqual(second_source.rows_written, 1)
            self.assertEqual(second_source.validation_errors, 1)

            report_payload = json.loads(Path(first_source.report_path).read_text(encoding="utf-8"))
            self.assertEqual(len(report_payload["validation_issues"]), 1)

    def test_rerun_is_idempotent_for_same_input_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "metrics.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "timestamp,service_name,host_id,metric_name,metric_value,unit,event_type,event_summary,anomaly_label",
                        "2026-04-14T10:00:00Z,edge-api,host-01,cpu_pct,71.4,percent,metric,,normal",
                    ]
                ),
                encoding="utf-8",
            )

            pipeline = IngestionPipeline(root)
            first_summary = pipeline.ingest([csv_path])
            second_summary = pipeline.ingest([csv_path])

            self.assertEqual(first_summary.sources[0].rows_written, 1)
            self.assertEqual(second_summary.sources[0].status, "skipped_existing_checksum")

            curated_path = root / "data" / "processed" / "curated" / "telemetry_records.csv"
            with curated_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)

    def test_ingests_sqlite_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            db_path = root / "telemetry.db"
            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE telemetry (
                        timestamp TEXT,
                        service_name TEXT,
                        host_id TEXT,
                        metric_name TEXT,
                        metric_value REAL,
                        unit TEXT,
                        event_type TEXT,
                        event_summary TEXT,
                        anomaly_label TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO telemetry VALUES
                    ('2026-04-14T10:00:00Z', 'edge-api', 'host-01', 'cpu_pct', 54.2, 'percent', 'metric', '', 'normal')
                    """
                )
                connection.commit()
            finally:
                connection.close()

            pipeline = IngestionPipeline(root)
            summary = pipeline.ingest([db_path], sql_table="telemetry")
            self.assertEqual(summary.sources[0].rows_written, 1)

            curated_path = root / "data" / "processed" / "curated" / "telemetry_records.csv"
            with curated_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["metric_name"], "cpu_pct")


if __name__ == "__main__":
    unittest.main()
