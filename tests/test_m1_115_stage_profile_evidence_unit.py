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
    / "M1_115_STAGE_PROFILE_CLOSURE_20260729"
    / "qualification.json"
)


class M115StageProfileEvidenceUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_evidence_is_bound_to_the_qualified_source_and_binaries(self):
        self.assertEqual(
            self.report["schema"],
            "bi100-m1-115-stage-profile-qualification-v1",
        )
        self.assertTrue(self.report["qualified"])
        self.assertEqual(
            self.report["source"]["commit"],
            "0d0a55918e2c39fc4de0cb7c7e609823d54679e1",
        )
        self.assertFalse(
            self.report["runtime"]["full_model_tp4_service_used"])
        for digest in self.report["extensions"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        for digest in self.report["artifact_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_all_component_and_lifecycle_gates_passed(self):
        self.assertEqual(self.report["runtime"]["runner_returncode"], 0)
        self.assertEqual(set(self.report["gates"].values()), {0})
        self.assertEqual(len(self.report["cases"]), 4)
        for case in self.report["cases"]:
            self.assertGreater(case["event_total_median_ms"], 0)
            self.assertLess(case["event_perturbation"], 0.01)

    def test_235k_profile_prioritizes_qk_and_pv_not_gather(self):
        case = next(
            row
            for row in self.report["cases"]
            if row["case"] == "production_235k_q5616"
        )
        shares = case["selected_stage_share"]
        self.assertGreater(shares["qk"] + shares["pv"], 0.73)
        self.assertLess(shares["gather"], 0.01)
        self.assertEqual(case["dominant_stage"], "pv")

    def test_promotion_boundary_remains_closed(self):
        decision = self.report["decision"]
        self.assertTrue(
            decision["deeper_fusion_design_selection_authorized"])
        self.assertFalse(
            decision["full_model_tp4_service_experiment_authorized"])
        self.assertFalse(decision["main_or_yaml_change_authorized"])
        self.assertFalse(decision["official_score_claim_authorized"])


if __name__ == "__main__":
    unittest.main()
