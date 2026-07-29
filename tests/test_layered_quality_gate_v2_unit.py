from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from tests import classify_teacher_forced_distribution as classifier


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "quality/layered_quality_gate.v2.json").read_text(
        encoding="utf-8"))


def _comparison(mode: str) -> dict:
    return {
        "schema": "bi100-teacher-forced-topk-comparison-v1",
        "version": 1,
        "status": "pass",
        "qualified": True,
        "comparison_mode": mode,
        "source_revision": "a" * 40,
        "instance": "private-instance",
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "top_k": 5,
        "case_count": 5,
        "sampled_positions": 320,
        "metrics": {
            "top1_agreement": 1.0,
            "top1_mismatch_count": 0,
            "mutually_uncovered_top1_mismatches": 0,
            "high_margin_guard_nats": 0.05,
            "high_margin_top1_mismatches": 0,
            "teacher_token_logprob_absolute_delta_max": 0.0,
            "teacher_token_logprob_absolute_delta_p99": 0.0,
            "shared_topk_logprob_absolute_delta_p99": 0.0,
            "mean_teacher_token_nll_regression": 0.0,
        },
        "cases": [
            {
                "id": f"length_{length}",
                "prompt_tokens": length,
                "sampled_positions": 64,
                "top1_agreement": 1.0,
                "mutually_uncovered_top1_mismatches": 0,
            }
            for length in (4096, 32768, 65536, 131072, 235000)
        ],
        "thresholds": {},
        "validation_reasons": [],
        "reasons": [],
        "authorization": {
            "teacher_forced_numerical_screen_authorized": True,
            "overall_promotion_authorized": False,
        },
        "privacy": {
            "contains_private_token_identity": False,
            "contains_token_ids": False,
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_credentials": False,
        },
    }


class LayeredQualityGateV2Tests(unittest.TestCase):

    def test_contract_separates_numeric_drift_and_capability_roles(self) -> None:
        self.assertEqual(
            CONTRACT["evidence_roles"]["operator_shadow_reference"],
            "hard_numeric")
        self.assertEqual(
            CONTRACT["evidence_roles"]["teacher_forced_distribution"],
            "equivalence_or_escalation")
        self.assertFalse(CONTRACT["promotion"][
            "cross_arm_token_identity_alone_can_reject"])
        self.assertFalse(CONTRACT["promotion"][
            "semantic_evidence_can_hide_operator_failure"])

    def test_tight_equivalence_still_needs_other_layers(self) -> None:
        result = classifier.classify(
            _comparison("candidate"),
            _comparison("control-repeat"),
            CONTRACT,
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            result["classification"], "tight-distribution-equivalence")
        self.assertFalse(result["promotion_authorized"])
        self.assertFalse(result["operator_numerics_decided"])

    def test_m1_132_scale_drift_escalates_instead_of_numeric_fail(self) -> None:
        candidate = _comparison("candidate")
        candidate["status"] = "fail"
        candidate["qualified"] = False
        candidate["metrics"].update({
            "top1_agreement": 0.940625,
            "top1_mismatch_count": 19,
            "mutually_uncovered_top1_mismatches": 3,
            "teacher_token_logprob_absolute_delta_max": 9.7978468,
            "teacher_token_logprob_absolute_delta_p99": 7.6840625,
            "shared_topk_logprob_absolute_delta_p99": 4.8989997,
            "mean_teacher_token_nll_regression": 0.10350425,
        })
        for case, agreement, uncovered in zip(
            candidate["cases"],
            (0.953125, 0.9375, 1.0, 0.921875, 0.890625),
            (0, 1, 0, 2, 0),
        ):
            case["top1_agreement"] = agreement
            case["mutually_uncovered_top1_mismatches"] = uncovered
        result = classifier.classify(
            candidate, _comparison("control-repeat"), CONTRACT)
        self.assertEqual(result["status"], "escalate")
        self.assertFalse(result["operator_numerics_decided"])
        self.assertFalse(result["capability_noninferiority_decided"])
        self.assertIn(
            "same-real-activation-operator-shadow-reference",
            result["next_required_evidence"])

    def test_bad_control_repeat_invalidates_attribution(self) -> None:
        repeat = _comparison("control-repeat")
        repeat["metrics"]["top1_agreement"] = 0.9875
        repeat["metrics"]["top1_mismatch_count"] = 4
        repeat["cases"][0]["top1_agreement"] = 0.9375
        result = classifier.classify(
            _comparison("candidate"), repeat, CONTRACT)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(
            result["classification"], "measurement-not-repeatable")

    def test_invalid_identity_is_not_reclassified_as_drift(self) -> None:
        candidate = _comparison("candidate")
        candidate["gpu_count"] = 3
        result = classifier.classify(
            candidate, _comparison("control-repeat"), CONTRACT)
        self.assertEqual(result["classification"], "invalid")

    def test_malformed_case_and_contract_do_not_raise(self) -> None:
        candidate = _comparison("candidate")
        candidate["cases"][0] = None
        contract = copy.deepcopy(CONTRACT)
        contract["teacher_forced_distribution"]["tight_equivalence"].pop(
            "minimum_top1_agreement")
        result = classifier.classify(
            candidate, _comparison("control-repeat"), contract)
        self.assertEqual(result["classification"], "invalid")
        self.assertTrue(result["validation_reasons"])

    def test_out_of_domain_metrics_are_invalid(self) -> None:
        for field, value in (
            ("top1_agreement", 2.0),
            ("top1_mismatch_count", -1),
            ("teacher_token_logprob_absolute_delta_p99", -0.1),
        ):
            with self.subTest(field=field):
                candidate = _comparison("candidate")
                candidate["metrics"][field] = value
                result = classifier.classify(
                    candidate, _comparison("control-repeat"), CONTRACT)
                self.assertEqual(result["classification"], "invalid")

    def test_inputs_are_not_mutated(self) -> None:
        candidate = _comparison("candidate")
        repeat = _comparison("control-repeat")
        before_candidate = copy.deepcopy(candidate)
        before_repeat = copy.deepcopy(repeat)
        classifier.classify(candidate, repeat, CONTRACT)
        self.assertEqual(candidate, before_candidate)
        self.assertEqual(repeat, before_repeat)


if __name__ == "__main__":
    unittest.main()
