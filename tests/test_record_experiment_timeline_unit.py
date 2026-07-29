from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "record_experiment_timeline.py"
SPEC = importlib.util.spec_from_file_location("experiment_timeline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExperimentTimelineTest(unittest.TestCase):

    def test_parallel_stages_report_wall_savings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.jsonl"
            MODULE.append_event(
                path, run_id="m1-140", stage="gpu0", event="start",
                wall_time_ns=1_000_000_000, monotonic_ns=1_000_000_000,
                pid=10)
            MODULE.append_event(
                path, run_id="m1-140", stage="gpu1", event="start",
                wall_time_ns=1_000_000_000, monotonic_ns=1_000_000_000,
                pid=11)
            MODULE.append_event(
                path, run_id="m1-140", stage="gpu0", event="end",
                status="pass", wall_time_ns=3_000_000_000,
                monotonic_ns=3_000_000_000, pid=10)
            MODULE.append_event(
                path, run_id="m1-140", stage="gpu1", event="end",
                status="pass", wall_time_ns=4_000_000_000,
                monotonic_ns=4_000_000_000, pid=11)
            report = MODULE.summarize(
                path, expected_run_id="m1-140")
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["wall_span_s"], 3.0)
        self.assertEqual(report["summed_stage_s"], 5.0)
        self.assertAlmostEqual(report["effective_parallelism"], 5.0 / 3.0)

    def test_missing_end_and_failed_stage_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.jsonl"
            MODULE.append_event(
                path, run_id="m1-140", stage="compile", event="start",
                wall_time_ns=1, monotonic_ns=1, pid=10)
            report = MODULE.summarize(path)
        self.assertFalse(report["qualified"])
        self.assertIn("stage compile has no end event", report["reasons"])

    def test_names_and_event_status_are_restricted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.jsonl"
            with self.assertRaises(ValueError):
                MODULE.append_event(
                    path, run_id="bad run", stage="compile", event="start")
            with self.assertRaises(ValueError):
                MODULE.append_event(
                    path, run_id="run", stage="compile", event="end")


if __name__ == "__main__":
    unittest.main()
