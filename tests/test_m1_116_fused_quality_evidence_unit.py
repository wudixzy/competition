from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs"
    / "experiments"
    / "evidence"
    / "M1_116_FUSED_PREFILL_QUALITY_ADJUDICATION_20260729"
    / "qualification.json"
)


class M116FusedQualityEvidenceUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_evidence_is_bound_to_the_exact_tp4_run(self):
        self.assertEqual(
            self.report["schema"],
            "bi100-m1-116-fused-prefill-quality-adjudication-v1",
        )
        self.assertEqual(
            self.report["source_revision"],
            "6eeb65a25ed5475ab0b9f2f6a965a21d707ff89f",
        )
        self.assertEqual(self.report["instance"], "ssh-73ca29ba")
        self.assertEqual(
            self.report["fixed_order"], ["fused-off", "fused-on"])

    def test_both_arms_pass_independent_quality_and_lifecycle_gates(self):
        for arm_name in ("control", "candidate"):
            arm = self.report[arm_name]
            self.assertEqual(arm["quality"]["passed"], 53)
            self.assertEqual(arm["quality"]["failed"], 0)
            self.assertEqual(arm["agent"]["passed"], 11)
            self.assertEqual(arm["agent"]["failed"], 0)
            self.assertEqual(arm["arm_returncode"], 0)
            self.assertTrue(arm["all_lifecycle_gates_passed"])
            self.assertTrue(arm["output_diagnostic"]["qualified"])
            self.assertTrue(
                arm["output_diagnostic"]["prefix_65k_cold_warm_exact"])
            self.assertTrue(
                arm["output_diagnostic"]["prefix_235k_cold_warm_exact"])
            self.assertTrue(arm["expected_4xx"]["qualified"])
            self.assertEqual(arm["expected_4xx"]["attribution_delta"], 0)

    def test_output_divergence_is_not_relabelled_as_exact(self):
        comparison = self.report["comparison"]
        self.assertEqual(comparison["quality_cases_qualified"], 52)
        self.assertEqual(comparison["quality_cases_failed"], 1)
        self.assertEqual(
            comparison["failed_quality_case"], "multimodal_input")
        self.assertTrue(comparison["diagnostic_valid"])
        self.assertTrue(comparison["next_token_exact"])
        self.assertFalse(comparison["strict_output_exact"])
        self.assertEqual(comparison["first_divergent_max_tokens"], 8)
        self.assertTrue(comparison["quality_adjudication_required"])
        self.assertEqual(comparison["runner_returncode"], 1)

    def test_decode_observation_stays_above_the_output_gate(self):
        control = self.report["control"]["exact_output_truncation"]
        candidate = self.report["candidate"]["exact_output_truncation"]
        self.assertGreaterEqual(control["derived_output_tps"], 20.0)
        self.assertGreaterEqual(candidate["derived_output_tps"], 20.0)
        self.assertGreaterEqual(
            candidate["relative_tps_delta_percent"], -2.0)

    def test_promotion_and_sensitive_evidence_remain_closed(self):
        authorization = self.report["authorization"]
        self.assertTrue(authorization["m1_117_long_context_adjudication"])
        for name in (
            "performance_authorized",
            "default_policy_change_authorized",
            "yaml_change_authorized",
            "main_merge_authorized",
            "production_promotion_authorized",
        ):
            self.assertFalse(authorization[name])
        self.assertEqual(set(self.report["privacy"].values()), {False})
        serialized = json.dumps(self.report, sort_keys=True).lower()
        for forbidden in (
            "request_content",
            "generated_output",
            "token_ids",
            "output_identities",
            "hmac_material",
            "cache_hashes",
            "credentials",
        ):
            self.assertIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
