from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_174_QUERY_TILED_REASSESSMENT_20260801")
SOURCE_REVISION = "1bdf7b826466ec7f0c98a20e8aa5d8d9391723d1"
CASES = (
    ("p90_total_16k_q8176", 1, 0.02120290922422004),
    ("p90_total_32k_q8176", 2, 0.016024077541737156),
    ("p90_total_64k_q8176", 3, 0.013404538450661948),
)


def load(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="ascii"))


class M1174QueryTiledReassessmentEvidenceTest(unittest.TestCase):
    def test_numeric_pass_does_not_hide_performance_failure(self):
        for case, gpu, speedup in CASES:
            with self.subTest(case=case):
                report = load(f"{case}.json")
                self.assertEqual(report["source_revision"], SOURCE_REVISION)
                self.assertEqual(report["visible_physical_gpu"], gpu)
                self.assertFalse(report["evaluation"]["qualified"])
                self.assertEqual(
                    report["evaluation"]["reasons"],
                    ["speedup is below the 0.98x cell floor"],
                )
                self.assertAlmostEqual(report["timings"]["speedup"], speedup)
                self.assertLess(report["timings"]["speedup"], 0.03)
                numerical = report["numerical"]
                calibrated = numerical["candidate_calibrated"]
                self.assertTrue(calibrated["candidate_finite"])
                self.assertLess(
                    calibrated[
                        "relative_l2_error_multiple_over_fp16_rounding"
                    ],
                    1.001,
                )
                self.assertLess(
                    calibrated[
                        "max_abs_error_multiple_over_fp16_rounding"
                    ],
                    1.001,
                )
                self.assertLess(
                    numerical["candidate_lse_relative_l2"], 3e-8)
                self.assertEqual(
                    numerical["candidate_repeat"],
                    {"lse_exact": True, "output_exact": True},
                )

    def test_aggregate_closes_every_later_stage(self):
        screen = load("screen.json")
        self.assertFalse(screen["qualified"])
        self.assertAlmostEqual(
            screen["median_speedup"], 0.016024077541737156)
        self.assertAlmostEqual(
            screen["minimum_speedup"], 0.013404538450661948)
        self.assertFalse(any(screen["authorization"].values()))
        self.assertEqual(len(screen["rows"]), 3)

        status = load("runner_status.json")
        self.assertEqual(status["source_revision"], SOURCE_REVISION)
        self.assertEqual(status["terminal_stage"], "paired_operator_cells")
        self.assertFalse(status["qualified"])
        self.assertTrue(all(status["lifecycle"].values()))
        self.assertFalse(any(status["authorization"].values()))
        self.assertEqual(
            load("failure.json")["stage"], "paired_operator_cells")
        self.assertTrue(load("fatal_scan.json")["qualified"])
        self.assertTrue(load("postflight_before.json")["qualified"])
        self.assertTrue(load("postflight_after.json")["qualified"])
        self.assertTrue(load("preflight_comparison.json")["qualified"])

    def test_identity_and_manifest_are_bound(self):
        identity = load("identity.json")
        self.assertEqual(identity["source_revision"], SOURCE_REVISION)
        self.assertEqual(identity["gpus"], [1, 2, 3])
        self.assertEqual(
            identity["baseline_extension_sha256"],
            "36e043f138aa87c635178e4aa6a30af710b87c3f3d7c2a3f1838fc0e365bd368",
        )
        self.assertEqual(
            identity["candidate_extension_sha256"],
            "ce75409a30b51e684f5384197b750952fdc63e9d19365f378791cf8ea3d3b67c",
        )
        self.assertEqual(
            identity["candidate_source_sha256"],
            "0217061a8803d2a181a01dd7316531d8cfed1fb84619d5f4e204acafe53b89c5",
        )

        manifest = (EVIDENCE / "SHA256SUMS").read_text(
            encoding="ascii").splitlines()
        expected_files = {
            path.name for path in EVIDENCE.glob("*.json")
        }
        self.assertEqual(
            {line.split("  ", 1)[1] for line in manifest}, expected_files)
        for line in manifest:
            expected, name = line.split("  ", 1)
            self.assertEqual(
                hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest(),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
