from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import unittest

import compare_m1_179_teacher_forced as comparison
from test_compare_teacher_forced_logprobs_v2_unit import _report


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((
    ROOT / "quality/layered_quality_gate.v2.json").read_text(
        encoding="utf-8"))


def _arm(label: str) -> dict:
    mode = "candidate" if label == "candidate" else "control"
    value = _report(mode)
    variant = comparison.EXPECTED_VARIANTS[label]
    value["schema"] = comparison.OBSERVATION_SCHEMA
    value["version"] = 2
    value["optimization"]["fused_prefill"] = "1"
    path = f"/tmp/{variant}.so"
    value["fused_variant"] = variant
    value["extension_identity"] = {
        "module_path": path,
        "runtime_loaded_module": path,
        "sha256": "a" * 64 if label != "candidate" else "b" * 64,
    }
    return value


class M1179ComparatorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.bootstrap_samples = comparison.BOOTSTRAP_SAMPLES
        comparison.BOOTSTRAP_SAMPLES = 500

    @classmethod
    def tearDownClass(cls) -> None:
        comparison.BOOTSTRAP_SAMPLES = cls.bootstrap_samples

    def test_correct_incremental_three_arm_binding_passes(self) -> None:
        result = comparison.compare(
            _arm("control_a"), _arm("control_b"), _arm("candidate"),
            CONTRACT)
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["arm_binding"]["control_a"],
                         "m1_109_fp32_qk")
        self.assertTrue(result["bootstrap"][
            "not_a_run_to_run_confidence_interval"])

    def test_fused_off_and_wrong_variants_are_invalid(self) -> None:
        for label in comparison.EXPECTED_VARIANTS:
            with self.subTest(label=label):
                arms = {name: _arm(name) for name in comparison.EXPECTED_VARIANTS}
                arms[label]["optimization"]["fused_prefill"] = "0"
                result = comparison.compare(
                    arms["control_a"], arms["control_b"], arms["candidate"],
                    CONTRACT)
                self.assertEqual(result["status"], "invalid", result)

                arms = {name: _arm(name) for name in comparison.EXPECTED_VARIANTS}
                arms[label]["fused_variant"] = "wrong"
                result = comparison.compare(
                    arms["control_a"], arms["control_b"], arms["candidate"],
                    CONTRACT)
                self.assertEqual(result["status"], "invalid", result)

    def test_token_cache_and_nonfinite_fail_closed(self) -> None:
        mutations = []
        token = _arm("candidate")
        token["cases"][0]["positions"][0]["actual_token_key"] = "f" * 64
        mutations.append(token)
        cached = _arm("candidate")
        cached["cases"][0]["cached_tokens"] = 1
        mutations.append(cached)
        nonfinite = _arm("candidate")
        nonfinite["cases"][0]["positions"][0]["top_logprobs"][0][
            "logprob"] = math.inf
        mutations.append(nonfinite)
        for candidate in mutations:
            result = comparison.compare(
                _arm("control_a"), _arm("control_b"), candidate, CONTRACT)
            self.assertEqual(result["status"], "invalid", result)

    def test_different_hmac_key_cannot_pair(self) -> None:
        candidate = _arm("candidate")
        for case in candidate["cases"]:
            for position in case["positions"]:
                translated = {}
                for item in position["top_logprobs"]:
                    old = item["token_key"]
                    translated[old] = f"{int(old, 16) + 100000:064x}"
                    item["token_key"] = translated[old]
                position["actual_token_key"] = translated[
                    position["actual_token_key"]]
        result = comparison.compare(
            _arm("control_a"), _arm("control_b"), candidate, CONTRACT)
        self.assertEqual(result["status"], "invalid", result)

    def test_aa_noise_raises_calibrated_margin_threshold(self) -> None:
        control_b = _arm("control_b")
        candidate = _arm("candidate")
        for value in (control_b, candidate):
            for position in value["cases"][0]["positions"]:
                for item in position["top_logprobs"]:
                    item["logprob"] -= 0.04
        result = comparison.compare(
            _arm("control_a"), control_b, candidate, CONTRACT)
        self.assertEqual(result["status"], "pass", result)
        self.assertAlmostEqual(result["thresholds"]["high_margin_nats"], 0.16)

    def test_candidate_high_margin_flip_is_incremental_drift(self) -> None:
        candidate = _arm("candidate")
        top = candidate["cases"][0]["positions"][0]["top_logprobs"]
        top[0], top[1] = top[1], top[0]
        top[0]["logprob"] = -0.05
        top[1]["logprob"] = -0.40
        result = comparison.compare(
            _arm("control_a"), _arm("control_b"), candidate, CONTRACT)
        self.assertEqual(result["status"], "inconclusive", result)
        self.assertEqual(result["classification"],
                         "incremental_fp16_qk_distribution_drift")

    def test_aa_high_margin_flip_blocks_incremental_attribution(self) -> None:
        control_b = _arm("control_b")
        top = control_b["cases"][0]["positions"][0]["top_logprobs"]
        top[0], top[1] = top[1], top[0]
        top[0]["logprob"] = -0.05
        top[1]["logprob"] = -0.40
        result = comparison.compare(
            _arm("control_a"), control_b, _arm("candidate"), CONTRACT)
        self.assertEqual(result["classification"],
                         "baseline_nondeterminism_or_measurement_noise")

    def test_aggregate_cancellation_cannot_hide_length_regression(self) -> None:
        candidate = _arm("candidate")
        for item in candidate["cases"][0]["positions"]:
            item["top_logprobs"][0]["logprob"] -= 0.2
        for item in candidate["cases"][-1]["positions"]:
            item["top_logprobs"][0]["logprob"] += 0.2
        result = comparison.compare(
            _arm("control_a"), _arm("control_b"), candidate, CONTRACT)
        self.assertAlmostEqual(
            result["incremental"]["paired_mean_nll_difference_nats"], 0.0)
        self.assertIn(4096, result["decision_basis"][
            "local_nll_regression_lengths"])
        self.assertEqual(result["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
