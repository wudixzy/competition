from __future__ import annotations

import json
from pathlib import Path
import statistics
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs"
    / "experiments"
    / "evidence"
    / "M1_114_CAPTURE_BOUNDARY_20260729"
    / "qualification.json"
)


class M114CaptureBoundaryEvidenceUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_targeted_capture_boundary_gate_passed(self):
        gate = self.report["capture_boundary_correctness"]
        self.assertTrue(gate["qualified"])
        self.assertTrue(gate["cold_warm_first_token_exact_in_every_arm"])
        self.assertTrue(gate["cold_warm_full_output_exact_in_every_arm"])
        self.assertEqual(gate["warm_cached_tokens"]["235000"], 234992)
        self.assertEqual(gate["warm_residual_prompt_tokens"]["235000"], 8)

    def test_fused_performance_is_repeatable_but_not_promoted(self):
        performance = self.report["fused_prefill_performance"]
        for improvements in performance[
                "cold_improvement_by_target"].values():
            self.assertEqual(len(improvements), 3)
            self.assertGreater(statistics.median(improvements), 0)
        self.assertGreater(
            statistics.median(
                performance["cold_improvement_by_target"]["235000"]),
            0.30,
        )
        quality = self.report["cross_selector_quality"]
        self.assertTrue(quality["first_generated_token_exact"])
        self.assertFalse(quality["all_full_output_hashes_exact"])
        self.assertFalse(self.report["qualified"])

    def test_lifecycle_gap_and_artifacts_are_explicit(self):
        lifecycle = self.report["lifecycle"]
        self.assertEqual(lifecycle["final_postflight"], 0)
        self.assertEqual(lifecycle["final_live_process_count"], 0)
        self.assertEqual(lifecycle["final_zombie_process_count"], 0)
        self.assertEqual(
            lifecycle["strict_no_recovery_qualification"], 1)
        for digest in self.report["artifact_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_promotion_boundary_remains_closed(self):
        decision = self.report["decision"]
        self.assertTrue(decision["focused_quality_adjudication_authorized"])
        for name in (
            "official_style_replay_authorized",
            "production_promotion_authorized",
            "main_merge_authorized",
            "yaml_change_authorized",
        ):
            self.assertFalse(decision[name])


if __name__ == "__main__":
    unittest.main()
