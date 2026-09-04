from __future__ import annotations

import json
from pathlib import Path
import unittest

from tests import classify_teacher_forced_distribution as legacy_classifier
from tests import validate_bi100_metrics_contract as metrics


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "quality/layered_quality_gate.v2.json").read_text())


class LayeredQualityGateV2Tests(unittest.TestCase):

    def test_contract_separates_numeric_distribution_and_capability(self) -> None:
        self.assertEqual(CONTRACT["evidence_roles"]["operator_numerics"],
                         "hard_calibrated_numeric")
        self.assertEqual(CONTRACT["evidence_roles"]["teacher_forced_distribution"],
                         "diagnostic_or_escalation")
        self.assertFalse(CONTRACT["promotion"][
            "semantic_evidence_can_hide_operator_failure"])

    def test_top1_has_no_universal_hard_floor(self) -> None:
        distribution = CONTRACT["teacher_forced_distribution"]
        self.assertIsNone(distribution["minimum_top1_agreement"])
        self.assertEqual(distribution["top1_agreement_role"], "diagnostic")

    def test_drift_uses_four_state_vocabulary(self) -> None:
        candidate = {
            "top1_agreement": 0.94,
            "mutual_topk_coverage": 0.99,
            "teacher_token_logprob_delta": 0.02,
            "shared_token_logprob_delta": 0.01,
            "paired_nll_difference": 0.02,
            "paired_nll_one_sided_95_upper_ci": 0.03,
            "first_divergent_token": 1,
            "baseline_top1_margin": 0.2,
            "high_margin_flips": 1,
        }
        result = metrics.classify_distribution(
            candidate,
            {"shared_logprob_delta_p99": 0.001,
             "paired_nll_upper_ci": 0.001},
            CONTRACT,
        )
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["classification"],
                         "distribution_drift_requires_adjudication")

    def test_july_v2_classifier_fails_closed_on_september_contract(self) -> None:
        result = legacy_classifier.classify({}, {}, CONTRACT)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["classification"], "invalid")


if __name__ == "__main__":
    unittest.main()
