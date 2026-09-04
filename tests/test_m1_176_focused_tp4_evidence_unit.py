from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((
    ROOT / "docs/experiments/evidence"
    / "M1_176_FP16_QK_FOCUSED_TP4_20260904/summary.json"
).read_text(encoding="utf-8"))


class M1176FocusedTp4EvidenceTests(unittest.TestCase):

    def test_valid_full_model_pair_passes_focused_gate(self) -> None:
        self.assertEqual(SUMMARY["status"], "pass")
        self.assertEqual(SUMMARY["change_scope"], "attention_operator")
        self.assertEqual(
            SUMMARY["model_path"],
            "/root/public-storage/models/Qwen/Qwen3.6-35B-A3B")
        self.assertGreaterEqual(
            SUMMARY["paired_performance"]["aggregate_gain"], 0.05)
        self.assertGreater(
            SUMMARY["paired_performance"]["one_sided_95_lower_ci"], 0.0)
        self.assertTrue(
            SUMMARY["paired_performance"]["all_buckets_stable"])

    def test_complete_two_arm_population_and_dispatch(self) -> None:
        for arm in ("control", "candidate"):
            self.assertEqual(SUMMARY["request_population"][arm], {
                "expected": 9, "attempted": 9, "completed": 9, "failed": 0})
        self.assertEqual(SUMMARY["dispatch"]["control"], 0)
        self.assertGreater(SUMMARY["dispatch"]["candidate"], 0)
        self.assertEqual(
            sorted(SUMMARY["raw_ttft_seconds"]["control"]),
            ["16384", "32768", "65536"])
        self.assertTrue(all(
            len(samples) == 3
            for arm in SUMMARY["raw_ttft_seconds"].values()
            for samples in arm.values()))

    def test_evidence_keeps_sha_and_scope_lean(self) -> None:
        self.assertFalse(SUMMARY["runtime"]["sha256_used"])
        self.assertIn("capability_strata", SUMMARY["not_run"])
        self.assertIn("teacher_forced_distribution", SUMMARY["not_run"])
        self.assertFalse(SUMMARY["promotion_authorized"])
        self.assertTrue(SUMMARY["lifecycle"]["postflight_qualified"])


if __name__ == "__main__":
    unittest.main()
