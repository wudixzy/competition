from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "quality" / "experiment_funnel.v1.json"
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_139_EXPERIMENT_FUNNEL_AUDIT_20260730.json"
)


class M1139ExperimentFunnelEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(EVIDENCE.read_text(encoding="ascii"))

    def test_frozen_contract_identity_matches_evidence(self):
        digest = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            self.report["new_funnel"]["contract_sha256"],
        )
        self.assertFalse(
            self.report["new_funnel"][
                "full_tp4_and_capability_gates_removed"
            ]
        )

    def test_cache_speedups_are_reproducible_from_recorded_times(self):
        pilot = self.report["pilot"]
        for prefix in ("compile", "overlay"):
            miss = pilot[f"actual_{prefix}_cache_miss_s"]
            hit = pilot[f"actual_{prefix}_cache_hit_s"]
            speedup = pilot[f"actual_{prefix}_cache_speedup"]
            self.assertGreater(miss, hit)
            self.assertTrue(math.isclose(
                miss / hit,
                speedup,
                rel_tol=1e-12,
                abs_tol=0.0,
            ))

    def test_partial_gpu_health_cannot_authorize_tp4_conclusions(self):
        pilot = self.report["pilot"]
        preflight = pilot["gpu_preflight"]
        results = preflight["parallel"]["gpu_results"]
        self.assertEqual(
            [row["gpu"] for row in results if row["ok"]],
            [2, 3],
        )
        self.assertFalse(preflight["parallel"]["qualified"])
        self.assertFalse(
            preflight["stale_probe_cleanup"]["sigkill_used"])
        after = preflight["post_cleanup_parallel"]
        self.assertEqual(
            [row["gpu"] for row in after["gpu_results"] if row["ok"]],
            [2, 3],
        )
        self.assertEqual(
            after["conclusion"],
            "stale_probe_not_root_cause",
        )
        self.assertFalse(pilot["tp4_conclusions_authorized"])
        self.assertIsNone(pilot["actual_capture_wall_s"])
        self.assertIsNone(pilot["actual_replay_wall_s"])
        self.assertFalse(any(self.report["privacy"].values()))


if __name__ == "__main__":
    unittest.main()
