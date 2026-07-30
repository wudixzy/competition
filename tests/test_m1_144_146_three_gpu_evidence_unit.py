from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs"
    / "experiments"
    / "evidence"
    / "M1_144_146_THREE_GPU_L1_20260730"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1144146ThreeGpuEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(
            (EVIDENCE / "aggregate.json").read_text(encoding="ascii"))

    def test_all_rotations_are_hash_bound_and_qualified(self) -> None:
        self.assertEqual(
            [run["gpu_order"] for run in self.value["runs"]],
            [[1, 2, 3], [2, 3, 1], [3, 1, 2]],
        )
        for run in self.value["runs"]:
            root = EVIDENCE / run["id"]
            self.assertTrue(run["qualified"])
            self.assertEqual(
                run["runner_status_sha256"],
                _sha256(root / "runner_status.json"),
            )
            self.assertEqual(
                run["screen_sha256"],
                _sha256(root / "screen.json"),
            )
            status = json.loads(
                (root / "runner_status.json").read_text(encoding="ascii"))
            self.assertTrue(status["qualified"])
            self.assertTrue(status["lifecycle"]["postflight_qualified"])
            self.assertTrue(
                status["lifecycle"]["preflight_comparison_qualified"])

    def test_numeric_screen_is_finite_and_replicated(self) -> None:
        self.assertTrue(self.value["all_finite"])
        self.assertTrue(self.value["all_runs_lifecycle_qualified"])
        for case in self.value["cases"].values():
            self.assertGreater(case["speedup_min"], 1.9)
            self.assertLess(case["output_relative_l2"], 1.0e-5)
            self.assertLess(case["lse_relative_l2"], 1.0e-6)
            self.assertLess(case["cross_card_spread_fraction"], 0.02)

    def test_three_gpu_result_does_not_authorize_l2_or_promotion(self) -> None:
        self.assertFalse(self.value["full_four_gpu_l1_contract_satisfied"])
        self.assertEqual(
            self.value["authorization"],
            {
                "four_gpu_l1_rerun_authorized": True,
                "l2_capture_authorized": False,
                "main_or_yaml_change_authorized": False,
                "official_score_claim_authorized": False,
            },
        )
        self.assertEqual(
            self.value["privacy"],
            {
                "contains_prompts": False,
                "contains_model_outputs": False,
                "contains_token_ids": False,
                "contains_credentials": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
