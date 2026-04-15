from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud_telemetry_intelligence_platform.serving.api import create_app
from cloud_telemetry_intelligence_platform.training.pipeline import TrainingPipeline


class ServingApiTests(unittest.TestCase):
    def test_serves_health_metrics_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feature_dir = root / "data" / "processed" / "features"
            feature_dir.mkdir(parents=True, exist_ok=True)
            feature_path = feature_dir / "window_features.csv"

            fieldnames = [
                "window_start",
                "window_minutes",
                "service_name",
                "host_id",
                "event_count",
                "event_frequency",
                "event_summaries",
                "is_anomaly",
                "target_next_latency_ms",
                "target_next_throughput_rps",
                "cpu_utilization",
                "cpu_utilization_was_imputed",
                "cpu_utilization_roll_mean",
                "cpu_utilization_roll_std",
                "cpu_utilization_roll_min",
                "cpu_utilization_roll_max",
                "cpu_utilization_roll_slope",
                "memory_utilization",
                "memory_utilization_was_imputed",
                "memory_utilization_roll_mean",
                "memory_utilization_roll_std",
                "memory_utilization_roll_min",
                "memory_utilization_roll_max",
                "memory_utilization_roll_slope",
                "latency_ms",
                "latency_ms_was_imputed",
                "latency_ms_roll_mean",
                "latency_ms_roll_std",
                "latency_ms_roll_min",
                "latency_ms_roll_max",
                "latency_ms_roll_slope",
                "throughput_rps",
                "throughput_rps_was_imputed",
                "throughput_rps_roll_mean",
                "throughput_rps_roll_std",
                "throughput_rps_roll_min",
                "throughput_rps_roll_max",
                "throughput_rps_roll_slope",
                "packet_drop_ratio",
                "packet_drop_ratio_was_imputed",
                "packet_drop_ratio_roll_mean",
                "packet_drop_ratio_roll_std",
                "packet_drop_ratio_roll_min",
                "packet_drop_ratio_roll_max",
                "packet_drop_ratio_roll_slope",
                "error_rate",
                "error_rate_was_imputed",
                "error_rate_roll_mean",
                "error_rate_roll_std",
                "error_rate_roll_min",
                "error_rate_roll_max",
                "error_rate_roll_slope",
                "request_failures",
                "request_failures_was_imputed",
                "request_failures_roll_mean",
                "request_failures_roll_std",
                "request_failures_roll_min",
                "request_failures_roll_max",
                "request_failures_roll_slope",
                "request_error_ratio",
                "cpu_throughput_imbalance",
                "short_term_latency_drift",
                "packet_drop_burst_count",
                "cpu_utilization_zscore",
                "memory_utilization_zscore",
                "latency_ms_zscore",
                "throughput_rps_zscore",
                "packet_drop_ratio_zscore",
                "error_rate_zscore",
                "request_failures_zscore",
            ]
            rows = []
            for index in range(12):
                anomaly = 1 if index in {3, 7, 10} else 0
                cpu = 0.35 + (0.06 * index) + (0.2 if anomaly else 0.0)
                latency = 85.0 + (4 * index) + (70 if anomaly else 0)
                throughput = 1200.0 - (15 * index) - (250 if anomaly else 0)
                packet_drop = 0.0 if not anomaly else 0.03
                error_rate = 0.01 if not anomaly else 0.28
                request_failures = 2.0 if not anomaly else 25.0
                next_latency = 85.0 + (4 * (index + 1)) + (70 if index + 1 in {3, 7, 10} else 0) if index < 11 else ""
                next_throughput = 1200.0 - (15 * (index + 1)) - (250 if index + 1 in {3, 7, 10} else 0) if index < 11 else ""
                rows.append(
                    {
                        "window_start": f"2026-04-14T10:{index:02d}:00+00:00",
                        "window_minutes": "1",
                        "service_name": "edge-api",
                        "host_id": "host-01",
                        "event_count": "1" if anomaly else "0",
                        "event_frequency": "1.0" if anomaly else "0.0",
                        "event_summaries": "packet loss burst" if anomaly else "",
                        "is_anomaly": str(anomaly),
                        "target_next_latency_ms": str(next_latency),
                        "target_next_throughput_rps": str(next_throughput),
                        "cpu_utilization": str(cpu),
                        "cpu_utilization_was_imputed": "0",
                        "cpu_utilization_roll_mean": str(cpu),
                        "cpu_utilization_roll_std": "0.01",
                        "cpu_utilization_roll_min": str(cpu - 0.02),
                        "cpu_utilization_roll_max": str(cpu + 0.02),
                        "cpu_utilization_roll_slope": "0.02",
                        "memory_utilization": str(0.4 + 0.03 * index),
                        "memory_utilization_was_imputed": "0",
                        "memory_utilization_roll_mean": str(0.4 + 0.03 * index),
                        "memory_utilization_roll_std": "0.01",
                        "memory_utilization_roll_min": str(0.38 + 0.03 * index),
                        "memory_utilization_roll_max": str(0.42 + 0.03 * index),
                        "memory_utilization_roll_slope": "0.01",
                        "latency_ms": str(latency),
                        "latency_ms_was_imputed": "0",
                        "latency_ms_roll_mean": str(latency - 5),
                        "latency_ms_roll_std": "2.0",
                        "latency_ms_roll_min": str(latency - 8),
                        "latency_ms_roll_max": str(latency + 3),
                        "latency_ms_roll_slope": "3.5",
                        "throughput_rps": str(throughput),
                        "throughput_rps_was_imputed": "0",
                        "throughput_rps_roll_mean": str(throughput + 10),
                        "throughput_rps_roll_std": "20.0",
                        "throughput_rps_roll_min": str(throughput - 15),
                        "throughput_rps_roll_max": str(throughput + 25),
                        "throughput_rps_roll_slope": "-10.0",
                        "packet_drop_ratio": str(packet_drop),
                        "packet_drop_ratio_was_imputed": "0",
                        "packet_drop_ratio_roll_mean": str(packet_drop),
                        "packet_drop_ratio_roll_std": "0.01",
                        "packet_drop_ratio_roll_min": "0.0",
                        "packet_drop_ratio_roll_max": str(packet_drop),
                        "packet_drop_ratio_roll_slope": str(packet_drop),
                        "error_rate": str(error_rate),
                        "error_rate_was_imputed": "0",
                        "error_rate_roll_mean": str(error_rate),
                        "error_rate_roll_std": "0.01",
                        "error_rate_roll_min": "0.0",
                        "error_rate_roll_max": str(error_rate),
                        "error_rate_roll_slope": str(error_rate),
                        "request_failures": str(request_failures),
                        "request_failures_was_imputed": "0",
                        "request_failures_roll_mean": str(request_failures),
                        "request_failures_roll_std": "1.0",
                        "request_failures_roll_min": str(request_failures),
                        "request_failures_roll_max": str(request_failures),
                        "request_failures_roll_slope": "1.0",
                        "request_error_ratio": str(error_rate),
                        "cpu_throughput_imbalance": str(cpu / throughput),
                        "short_term_latency_drift": str(5.0 if anomaly else -1.0),
                        "packet_drop_burst_count": "1" if anomaly else "0",
                        "cpu_utilization_zscore": "0.0",
                        "memory_utilization_zscore": "0.0",
                        "latency_ms_zscore": "0.0",
                        "throughput_rps_zscore": "0.0",
                        "packet_drop_ratio_zscore": "0.0",
                        "error_rate_zscore": "0.0",
                        "request_failures_zscore": "0.0",
                    }
                )

            with feature_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            TrainingPipeline(root).train()
            client = TestClient(create_app(root))

            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.json()["models_loaded"])

            metrics = client.get("/metrics")
            self.assertEqual(metrics.status_code, 200)
            self.assertIn("classification_models", metrics.json())

            payload = {
                "service_name": "edge-api",
                "host_id": "host-01",
                "window_start": "2026-04-14T10:12:30Z",
                "window_minutes": 1,
                "rolling_window_size": 3,
                "history": [
                    {"cpu_utilization": 0.7, "latency_ms": 120.0, "throughput_rps": 900.0, "error_rate": 0.02},
                    {"cpu_utilization": 0.82, "latency_ms": 150.0, "throughput_rps": 780.0, "error_rate": 0.18},
                ],
                "observations": [
                    {"timestamp": "2026-04-14T10:12:05Z", "metric_name": "cpu_pct", "metric_value": 88.0, "unit": "percent"},
                    {"timestamp": "2026-04-14T10:12:10Z", "metric_name": "latency_ms", "metric_value": 175.0, "unit": "ms"},
                    {"timestamp": "2026-04-14T10:12:20Z", "metric_name": "throughput_rps", "metric_value": 650.0, "unit": "rps"},
                    {"timestamp": "2026-04-14T10:12:30Z", "metric_name": "request_failures", "metric_value": 20.0, "unit": "count", "event_type": "event", "event_summary": "timeout spike"},
                    {"timestamp": "2026-04-14T10:12:35Z", "metric_name": "packet_drop_pct", "metric_value": 3.0, "unit": "percent", "event_type": "event", "event_summary": "packet loss burst"},
                ],
            }

            anomaly = client.post("/predict/anomaly", json=payload)
            self.assertEqual(anomaly.status_code, 200)
            anomaly_body = anomaly.json()
            self.assertEqual(anomaly_body["service_name"], "edge-api")
            self.assertGreaterEqual(len(anomaly_body["predictions"]), 1)
            self.assertIn("cpu_utilization", anomaly_body["feature_preview"])

            regression = client.post("/predict/regression", json=payload)
            self.assertEqual(regression.status_code, 200)
            self.assertGreaterEqual(len(regression.json()["predictions"]), 1)

            batch = client.post("/predict/batch", json=[payload, payload])
            self.assertEqual(batch.status_code, 200)
            self.assertEqual(len(batch.json()["results"]), 2)

            prediction_log = root / "data" / "serving" / "predictions.jsonl"
            self.assertTrue(prediction_log.exists())
            lines = prediction_log.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 4)
            parsed = [json.loads(line) for line in lines]
            self.assertEqual(parsed[0]["service_name"], "edge-api")

