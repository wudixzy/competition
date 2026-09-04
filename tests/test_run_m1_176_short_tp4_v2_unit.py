from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import run_m1_176_short_tp4_v2 as runner


def _replay() -> dict:
    return {
        "schema": "bi100-m1-176-four-rank-real-activation-replay-v2",
        "version": 2, "qualified": True, "result_status": "pass",
        "terminal_stage": "parallel_four_rank_replay",
        "source_revision": "a" * 40, "runtime_identity": "overlay",
        "authorization": {"l3_short_tp4_authorized": True,
                          "long_context_or_formal_score_authorized": False},
        "aggregate": {
            "four_rank_replay_complete": True, "g2_reasons": [],
            "invalid_reasons": [],
            "rows": [{"logical_tp_rank": rank, "all_g2_qualified": True,
                      "record_count": 3} for rank in range(4)],
        },
    }


def _capture() -> dict:
    return {
        "schema": "qwen36-diagnostic-service-gate-v2", "version": 2,
        "qualified": True,
        "workload_scope": "m1-176-activation-capture-only",
        "activation_capture_summary": {"request_count": 3},
    }


class ShortTp4V2RunnerTests(unittest.TestCase):

    def test_l3_requires_fixed_full_model_path(self) -> None:
        self.assertTrue(runner.is_expected_full_model_path(
            Path("/root/public-storage/models/Qwen/Qwen3.6-35B-A3B")))
        self.assertFalse(runner.is_expected_full_model_path(
            Path("/tmp/diagnostic-checkpoint/model")))
        self.assertFalse(runner.is_expected_full_model_path(
            Path("relative/model")))

    def test_service_environment_binds_teacher_keys(self) -> None:
        value = runner.ShortTp4V2Runner.__new__(runner.ShortTp4V2Runner)
        root = Path(__file__).resolve().parents[1]
        value.root = root
        value.runtime_site = root
        value.runtime_install = root / "README.md"
        value.run_root = Path("/tmp/m1-176-unit")
        value.model_path = root
        value.args = SimpleNamespace(selector="control_a")
        environment = value.service_environment()
        self.assertEqual(environment["BI100_GDN_CACHE_POLICY"], "admission64")
        self.assertEqual(environment["BI100_GDN_RESTORE_MODE"], "hybrid64")
        self.assertEqual(environment["BI100_KV_EVICTION_POLICY"], "lru")

    def test_l2_v2_authorizes_fixed_short_tp4_only(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            root.chmod(0o700)
            replay = root / "replay.json"
            capture = root / "capture.json"
            replay.write_text(json.dumps(_replay()), encoding="utf-8")
            capture.write_text(json.dumps(_capture()), encoding="utf-8")
            replay.chmod(0o600)
            capture.chmod(0o600)
            result = runner.validate_l2_authorization(replay, capture)
            self.assertTrue(result["replay_qualified"])
            self.assertEqual(result["four_rank_cells"], 12)

    def test_incomplete_rank_population_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            root.chmod(0o700)
            value = _replay()
            value["aggregate"]["rows"].pop()
            replay = root / "replay.json"
            capture = root / "capture.json"
            replay.write_text(json.dumps(value), encoding="utf-8")
            capture.write_text(json.dumps(_capture()), encoding="utf-8")
            replay.chmod(0o600)
            capture.chmod(0o600)
            with self.assertRaises(ValueError):
                runner.validate_l2_authorization(replay, capture)


if __name__ == "__main__":
    unittest.main()
