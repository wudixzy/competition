from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_m1_176_tp1_derived_replay.py"
SPEC = importlib.util.spec_from_file_location("m1_176_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC.loader.exec_module(MODULE)
finally:
    sys.path.remove(str(SCRIPT.parent))


class M1176FourRankReplayRunnerTest(unittest.TestCase):

    @staticmethod
    def _report(rank: int, qualified: bool = True):
        record = {
            "context_tokens": 24576,
            "query_length": 32,
            "candidate_numeric": {
                "relative_l2_error_ratio": 1.0,
                "maximum_absolute_error_ratio": 1.0,
                "candidate_lse_relative_l2": 1e-6,
            },
            "timing": {"order_balanced_geometric_speedup": 1.2},
        }
        return {
            "schema": "bi100-m1-176-tp1-derived-rank-replay-v2",
            "version": 2,
            "logical_tp_rank": rank,
            "visible_physical_gpu": rank,
            "capture_source_revision": "a" * 40,
            "baseline_source_revision": "a" * 40,
            "candidate_source_revision": "a" * 40,
            "runtime_identity": "overlay",
            "baseline_extension": {"sha256": "b" * 64},
            "candidate_extension": {"sha256": "c" * 64},
            "all_qualified": qualified,
            "records": [
                record,
                {**record, "context_tokens": 57344},
                {**record, "context_tokens": 122880},
            ],
        }

    def _aggregate(self, missing_rank=None, failed_rank=None):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            for rank in MODULE.RANKS:
                if rank == missing_rank:
                    continue
                (root / f"rank-{rank}-replay.json").write_text(
                    json.dumps(self._report(rank, rank != failed_rank)),
                    encoding="ascii")
            return MODULE._aggregate(
                root, [0, 1, 2, 3], "a" * 40, "overlay",
                "b" * 64, "c" * 64)

    def test_complete_four_rank_g2_pass(self):
        result = self._aggregate()
        self.assertTrue(result["qualified"])
        self.assertEqual(result["result_status"], "pass")
        self.assertTrue(result["four_rank_replay_complete"])
        self.assertFalse(result["tp4_model_execution_claimed"])

    def test_numeric_failure_is_fail_not_invalid(self):
        result = self._aggregate(failed_rank=2)
        self.assertFalse(result["qualified"])
        self.assertEqual(result["result_status"], "fail")
        self.assertTrue(result["four_rank_replay_complete"])
        self.assertEqual(result["invalid_reasons"], [])

    def test_missing_rank_is_invalid(self):
        result = self._aggregate(missing_rank=3)
        self.assertEqual(result["result_status"], "invalid")
        self.assertFalse(result["four_rank_replay_complete"])


if __name__ == "__main__":
    unittest.main()
