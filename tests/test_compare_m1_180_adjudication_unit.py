from __future__ import annotations

import copy
import math
import unittest

import compare_m1_180_adjudication as comparison
from test_compare_teacher_forced_logprobs_v2_unit import _report


def _teacher(label: str) -> dict:
    mode = "candidate" if label == "m1_162" else "control"
    value = _report(mode)
    value.update({"source_revision": "r", "runtime_identity": "runtime",
                  "instance": "instance", "model_path": "/model"})
    value["optimization"]["fused_prefill"] = "0" if label == "fused_off" else "1"
    if label != "fused_off":
        variant = comparison.ARMS[label]
        value["schema"] = "bi100-teacher-forced-topk-observation-v2"
        value["version"] = 2
        value["fused_variant"] = variant
        value["extension_identity"] = {
            "module_path": f"/tmp/{variant}.so",
            "runtime_loaded_module": f"/tmp/{variant}.so",
            "sha256": "a" * 64,
        }
    return value


def _performance() -> dict:
    cases = []
    for target in (16384, 32768, 65536):
        for repetition in range(2):
            cases.append({
                "target_prompt_tokens": target, "repetition": repetition,
                "response": {"ttft_s": 10.0, "cached_tokens": 0},
            })
    return {"cases": cases}


def _arm(label: str, *, complete: bool = True) -> dict:
    limit = 10 if complete else 4
    cases = []
    for stratum in comparison.STRATA:
        for ordinal in range(limit):
            cases.append({
                "case_id": f"{stratum}_{ordinal:02d}", "stratum": stratum,
                "ordinal": ordinal,
                "stage": "smoke" if ordinal < 4 else "extended",
                "pass": True, "validator": "fixed", "http_status": 200,
                "response_contract_complete": True, "finish_reason": "stop",
                "prompt_tokens": 100, "cached_tokens": 0,
                "completion_tokens": 2, "elapsed_s": 1.0,
                "all_values_finite": True,
                **({"reasoning_protocol_valid": True}
                   if stratum == "reasoning" else {}),
            })
    teacher = _teacher(label) if complete else None
    performance = (_performance() if label in ("m1_109", "m1_162")
                   and complete else None)
    extra = (4 if teacher else 0) + (6 if performance else 0)
    return {
        "schema": "bi100-m1-180-arm-observation-v1", "version": 1,
        "arm": label, "algorithm_variant": comparison.ARMS[label],
        "source_revision": "r", "runtime_identity": "runtime",
        "instance": "instance", "model_path": "/model", "workload_id": "w",
        "capability": {
            "strata": list(comparison.STRATA), "smoke_per_stratum": 4,
            "full_per_stratum": 10, "cases": cases,
            "smoke_completed": 24, "extended_triggered": complete,
            "critical_smoke_baseline_only": [], "complete": complete,
        },
        "teacher_forced": teacher, "performance": performance,
        "request_population": {"attempted": len(cases) + extra,
                               "completed": len(cases) + extra, "failed": 0},
    }


AA = {
    "top1_flip_count": 0,
    "shared_token_abs_logprob_delta_p99_nats": 0.0,
    "position_sampling_one_sided_95_upper_nats": 0.0,
}


class M1180ComparatorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = comparison.BOOTSTRAP_SAMPLES
        comparison.BOOTSTRAP_SAMPLES = 200
        import compare_m1_179_teacher_forced as m179
        cls.distribution_samples = m179.BOOTSTRAP_SAMPLES
        m179.BOOTSTRAP_SAMPLES = 200

    @classmethod
    def tearDownClass(cls) -> None:
        comparison.BOOTSTRAP_SAMPLES = cls.samples
        import compare_m1_179_teacher_forced as m179
        m179.BOOTSTRAP_SAMPLES = cls.distribution_samples

    def test_three_arm_screen_is_valid_but_not_promotion_pass(self) -> None:
        result = comparison.compare(
            _arm("fused_off"), _arm("m1_109"), _arm("m1_162"), AA)
        self.assertEqual(result["status"], "inconclusive", result)
        self.assertEqual(result["capability"]["m1_109_vs_m1_162"]
                         ["development_screen_status"], "inconclusive")
        self.assertTrue(result["capability"]["m1_109_vs_m1_162"]
                        ["statistical_strata_underpowered"])
        self.assertEqual(result["capability"]["m1_109_vs_m1_162"]
                         ["deterministic_contracts"]["status"], "pass")

    def test_capability_pair_reports_all_four_cells_and_strata(self) -> None:
        candidate = _arm("m1_162")
        next(case for case in candidate["capability"]["cases"]
             if case["case_id"] == "tools_00")["pass"] = False
        result = comparison.compare(
            _arm("fused_off"), _arm("m1_109"), candidate, AA)
        pair = result["capability"]["m1_109_vs_m1_162"]
        self.assertEqual(result["status"], "fail")
        self.assertEqual(pair["baseline_only"], 1)
        self.assertEqual(pair["strata"]["code"]["sample_count"], 10)
        self.assertIn("two_sided_exact_p", pair["exact_mcnemar"])

    def test_smoke_regression_is_valid_early_capability_failure(self) -> None:
        candidate = _arm("m1_162", complete=False)
        next(case for case in candidate["capability"]["cases"]
             if case["case_id"] == "tools_00")["pass"] = False
        result = comparison.compare(
            _arm("fused_off"), _arm("m1_109"), candidate, AA)
        self.assertEqual(result["status"], "fail", result)
        self.assertEqual(result["incremental_performance"]["status"], "not_run")

    def test_wrong_variant_token_cache_and_nonfinite_are_invalid(self) -> None:
        mutations = []
        wrong = _arm("m1_162")
        wrong["algorithm_variant"] = "m1_109_fp32_qk"
        mutations.append(wrong)
        cached = _arm("m1_162")
        cached["teacher_forced"]["cases"][0]["cached_tokens"] = 1
        mutations.append(cached)
        nonfinite = _arm("m1_162")
        nonfinite["teacher_forced"]["cases"][0]["positions"][0][
            "top_logprobs"][0]["logprob"] = math.inf
        mutations.append(nonfinite)
        for candidate in mutations:
            with self.subTest():
                result = comparison.compare(
                    _arm("fused_off"), _arm("m1_109"), candidate, AA)
                self.assertEqual(result["status"], "invalid", result)

    def test_different_teacher_identity_cannot_pair(self) -> None:
        candidate = _arm("m1_162")
        candidate["teacher_forced"]["cases"][0]["positions"][0][
            "actual_token_key"] = "f" * 64
        result = comparison.compare(
            _arm("fused_off"), _arm("m1_109"), candidate, AA)
        self.assertEqual(result["status"], "invalid", result)

    def test_performance_is_direct_m1_109_to_m1_162(self) -> None:
        candidate = _arm("m1_162")
        for case in candidate["performance"]["cases"]:
            case["response"]["ttft_s"] = 8.0
        result = comparison.compare(
            _arm("fused_off"), _arm("m1_109"), candidate, AA)
        self.assertAlmostEqual(
            result["incremental_performance"]["paired_mean_gain"], 0.25)
        self.assertEqual(result["incremental_performance"]["status"],
                         "inconclusive")
        self.assertEqual(result["incremental_performance"]["classification"],
                         "positive_diagnostic_underpowered")

    def test_aa_is_bound_only_to_matching_left_control(self) -> None:
        result = comparison.compare(
            _arm("fused_off"), _arm("m1_109"), _arm("m1_162"), AA)
        fused = result["distribution"]["fused_off_vs_m1_109"]
        incremental = result["distribution"]["m1_109_vs_m1_162"]
        self.assertFalse(fused["calibrated"])
        self.assertEqual(fused["classification"],
                         "uncalibrated_distribution_diagnostic")
        self.assertTrue(incremental["calibrated"])
        self.assertEqual(incremental["aa_control_variant"],
                         "m1_109_fp32_qk")

    def test_deterministic_stratum_does_not_use_statistical_floor(self) -> None:
        pair = comparison.capability_pair(_arm("m1_109"), _arm("m1_162"),
                                          "m1_109_vs_m1_162")
        tools = pair["strata"]["tools"]
        self.assertEqual(tools["gate_type"], "deterministic_contract")
        self.assertEqual(tools["status"], "pass")
        self.assertNotIn("underpowered_for_stratum_promotion", tools)


if __name__ == "__main__":
    unittest.main()
