from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_173_BATCHED_SPLIT_PV_20260801")
CASES = (
    "p90_total_16k_q8176",
    "p90_total_32k_q8176",
    "p90_total_64k_q8176",
)
SOURCE_REVISION = "b3d20731d31a8b67d1ec44bc1725d34fd2752508"


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="ascii"))


class M1173BatchedSplitPvEvidenceTest(unittest.TestCase):
    def test_screen_closes_only_the_performance_route(self):
        screen = load("screen.json")
        self.assertFalse(screen["qualified"])
        self.assertEqual(
            screen["reasons"], ["median speedup is below 1.08x"])
        self.assertAlmostEqual(screen["median_speedup"], 1.0043696732129823)
        self.assertGreaterEqual(screen["minimum_speedup"], 0.98)
        self.assertEqual([row["case"] for row in screen["rows"]], list(CASES))
        self.assertTrue(all(row["qualified"] for row in screen["rows"]))
        self.assertTrue(all(
            row["candidate_baseline_output_l2"] == 0.0
            for row in screen["rows"]
        ))
        self.assertFalse(any(screen["authorization"].values()))

    def test_cells_pass_numeric_repeat_and_identity_gates(self):
        for case in CASES:
            with self.subTest(case=case):
                report = load(f"{case}.json")
                self.assertEqual(report["source_revision"], SOURCE_REVISION)
                self.assertTrue(report["evaluation"]["qualified"])
                self.assertEqual(
                    report["baseline_extension"]["module_name"],
                    "corex_fused_paged_prefill_fp16_qk",
                )
                self.assertEqual(
                    report["candidate_extension"]["module_name"],
                    "corex_fused_paged_prefill_batched_split_pv",
                )
                numerical = report["numerical"]
                self.assertTrue(
                    numerical["candidate_calibrated"]["candidate_finite"])
                self.assertLessEqual(
                    numerical["candidate_calibrated"][
                        "relative_l2_error_multiple_over_fp16_rounding"],
                    2.0,
                )
                self.assertLessEqual(
                    numerical["candidate_calibrated"][
                        "max_abs_error_multiple_over_fp16_rounding"],
                    2.0,
                )
                self.assertEqual(
                    numerical["candidate_vs_baseline"]["output_relative_l2"],
                    0.0,
                )
                self.assertEqual(
                    numerical["candidate_vs_baseline"]["lse_relative_l2"],
                    0.0,
                )
                self.assertTrue(
                    numerical["candidate_repeat"]["output_exact"])
                self.assertTrue(numerical["candidate_repeat"]["lse_exact"])

    def test_lifecycle_is_clean_and_failure_is_fail_closed(self):
        status = load("runner_status.json")
        self.assertEqual(status["source_revision"], SOURCE_REVISION)
        self.assertFalse(status["qualified"])
        self.assertTrue(all(
            row["returncode"] == 0 for row in status["cell_processes"]
        ))
        self.assertTrue(all(status["lifecycle"].values()))
        self.assertFalse(any(status["authorization"].values()))
        self.assertEqual(
            load("failure.json")["stage"], "paired_operator_cells")
        self.assertTrue(load("fatal_scan.json")["qualified"])
        self.assertTrue(load("postflight_before.json")["qualified"])
        self.assertTrue(load("postflight_after.json")["qualified"])
        self.assertTrue(load("preflight_comparison.json")["qualified"])

    def test_sha256_manifest_matches(self):
        manifest = (EVIDENCE / "SHA256SUMS").read_text(
            encoding="ascii").splitlines()
        self.assertTrue(manifest)
        for line in manifest:
            expected, name = line.split("  ", 1)
            payload = (EVIDENCE / name).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
