from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_175_FP16_QK_CROSS_INSTANCE_20260804")
HISTORICAL = (
    ROOT / "docs" / "experiments" / "evidence"
    / "M1_162_CALIBRATED_FP16_QK_20260730")
CASES = (
    "p90_total_16k_q8176",
    "p90_total_32k_q8176",
    "p90_total_64k_q8176",
)


def load(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="ascii"))


class M1175Fp16QkCrossInstanceEvidenceTest(unittest.TestCase):
    def test_fixed_cells_repeat_numeric_and_speed_result(self):
        for case in CASES:
            with self.subTest(case=case):
                current = load(EVIDENCE, f"{case}.json")
                historical = load(HISTORICAL, f"{case}.json")
                self.assertTrue(current["evaluation"]["qualified"])
                self.assertGreaterEqual(current["timings"]["speedup"], 1.15)
                speedup_change = abs(
                    current["timings"]["speedup"]
                    / historical["timings"]["speedup"]
                    - 1.0
                )
                self.assertLess(speedup_change, 0.002)
                numerical = current["numerical"]
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
                    numerical["candidate_lse_relative_l2"], 4e-8)
                self.assertEqual(
                    numerical["candidate_repeat"],
                    {"lse_exact": True, "output_exact": True},
                )

    def test_identity_preflight_and_authorization_fail_closed(self):
        identity = load(EVIDENCE, "identity.json")
        self.assertEqual(identity["instance"], "ssh-1f88d35a")
        self.assertEqual(identity["gpu"], 1)
        self.assertFalse(any(identity["authorization"].values()))
        self.assertEqual(
            identity["source_files"]["numeric_contract_sha256"],
            "1be3ccf34cef906fdc8345c1754960bb4485259f51c3963ab9ca15fd3a4bdb05",
        )
        for stage in ("preflight-before.json", "preflight-after.json"):
            preflight = load(EVIDENCE, stage)
            self.assertTrue(preflight["ok"])
            self.assertEqual(preflight["gpus"], [1])
            self.assertEqual(preflight["results"][0]["free"], 34057748480)
        self.assertTrue(load(EVIDENCE, "fatal_scan.json")["qualified"])

    def test_cross_instance_comparison_is_tight(self):
        comparison = load(EVIDENCE, "comparison.json")
        self.assertTrue(comparison["qualified"])
        self.assertLess(
            comparison["maximum_absolute_speedup_change_percent"], 0.2)
        self.assertGreaterEqual(comparison["new_median_speedup"], 1.16)
        self.assertEqual(
            [row["case"] for row in comparison["cases"]], list(CASES))

    def test_manifest_authenticates_every_json_file(self):
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
