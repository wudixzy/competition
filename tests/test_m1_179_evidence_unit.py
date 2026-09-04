from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((
    ROOT / "docs/experiments/evidence/"
    "M1_179_FP16_QK_INCREMENTAL_DISTRIBUTION_20260905/summary.json"
).read_text(encoding="utf-8"))


class M1179EvidenceTests(unittest.TestCase):

    def test_valid_aa_and_incremental_drift_are_separate(self) -> None:
        self.assertEqual(SUMMARY["experiment_validity"], "pass")
        self.assertEqual(SUMMARY["status"], "inconclusive")
        self.assertEqual(SUMMARY["classification"],
                         "incremental_fp16_qk_distribution_drift")
        self.assertEqual(SUMMARY["aa_distribution"]["top1_flip_count"], 0)
        self.assertEqual(SUMMARY["incremental_distribution"][
            "top1_flip_count"], 11)
        self.assertEqual(SUMMARY["incremental_distribution"][
            "high_margin_flip_count"], 8)

    def test_population_variant_and_privacy_contract(self) -> None:
        self.assertEqual(SUMMARY["request_population"][
            "total_teacher_forced_model_requests"], 12)
        self.assertTrue(SUMMARY["service_contract"][
            "all_arms_fused_prefill_enabled"])
        self.assertEqual(SUMMARY["service_contract"]["arm_variants"], {
            "control_a": "m1_109_fp32_qk",
            "candidate": "m1_162_fp16_qk",
            "control_b": "m1_109_fp32_qk",
        })
        self.assertTrue(SUMMARY["lifecycle"]["final_postflight_qualified"])
        self.assertTrue(all(value is False
                            for value in SUMMARY["privacy"].values()))


if __name__ == "__main__":
    unittest.main()
