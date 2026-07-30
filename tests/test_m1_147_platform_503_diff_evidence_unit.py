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
    / "M1_147_PLATFORM_503_DIFF_TRIAGE_20260730"
    / "result.json"
)


class M1147Platform503DiffEvidenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.value = json.loads(EVIDENCE.read_text(encoding="ascii"))

    def test_result_is_bound_to_both_revisions(self) -> None:
        self.assertEqual(
            self.value["platform_result"]["source_revision"],
            "503fa7c670b6172d9a3e2912166e78317f5e289f",
        )
        self.assertEqual(
            self.value["comparison"]["current_main_revision"],
            "fb0084fc778e62c26d6a6e108b87dc027ae2ed79",
        )
        self.assertTrue(
            self.value["comparison"]["computility_run_identical"])

    def test_diagnostic_does_not_overclaim_platform_fixes(self) -> None:
        diagnostic = self.value["current_diagnostic"]
        conclusions = self.value["conclusions"]
        self.assertFalse(diagnostic["full_model_evaluated"])
        self.assertFalse(diagnostic["performance_evaluated"])
        self.assertFalse(diagnostic["overall_qualified"])
        self.assertTrue(conclusions["n2_current_shape_works"])
        self.assertTrue(conclusions["base64_current_shape_works"])
        self.assertFalse(conclusions["post_platform_fix_proven_for_n2"])
        self.assertFalse(conclusions["post_platform_fix_proven_for_base64"])
        self.assertFalse(conclusions["platform_tool_4xx_resolved"])

    def test_no_promotion_is_authorized(self) -> None:
        self.assertEqual(
            self.value["authorization"],
            {
                "full_model_quality_authorized": False,
                "tp4_performance_authorized": False,
                "main_or_yaml_change_authorized": False,
            },
        )
        self.assertEqual(
            self.value["privacy"],
            {
                "contains_raw_requests": False,
                "contains_raw_model_outputs": False,
                "contains_token_ids": False,
                "contains_credentials": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
