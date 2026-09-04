from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from tests import validate_bi100_metrics_contract as metrics


ROOT = Path(__file__).resolve().parents[1]
LAYERED_V1 = json.loads((ROOT / "quality/layered_quality_gate.v1.json").read_text())
LAYERED_V2 = json.loads((ROOT / "quality/layered_quality_gate.v2.json").read_text())
FUNNEL_V1 = json.loads((ROOT / "quality/experiment_funnel.v1.json").read_text())
FUNNEL_V2 = json.loads((ROOT / "quality/experiment_funnel.v2.json").read_text())


def numeric() -> dict:
    return {
        "reference_finite": True,
        "baseline_finite": True,
        "rounded_reference_finite": True,
        "candidate_finite": True,
        "candidate_repeat_finite": True,
        "metadata_exact": True,
        "candidate_relative_l2": 2e-4,
        "baseline_relative_l2": 1e-4,
        "candidate_max_abs": 4e-3,
        "baseline_max_abs": 2e-3,
        "candidate_lse_relative_l2": 2e-6,
        "baseline_lse_relative_l2": 1e-6,
        "candidate_vs_rounded_relative_l2": 0.5,
        "candidate_vs_rounded_max_abs": 50.0,
    }


def distribution() -> dict:
    return {
        "top1_agreement": 0.10,
        "mutual_topk_coverage": 0.80,
        "teacher_token_logprob_delta": 0.005,
        "shared_token_logprob_delta": 0.004,
        "paired_nll_difference": 0.002,
        "paired_nll_one_sided_95_upper_ci": 0.009,
        "first_divergent_token": 7,
        "baseline_top1_margin": 0.01,
        "high_margin_flips": 0,
    }


def capability_stratum(*, underpowered: bool = False) -> dict:
    return {
        "sample_count": 10,
        "paired_results": {
            "both_pass": 8, "baseline_only": 0,
            "candidate_only": 1, "both_fail": 1,
        },
        "paired_lower_ci": -0.01,
        "paired_bootstrap_reported": True,
        "exact_mcnemar_reported": True,
        "underpowered": underpowered,
    }


class BI100MetricsContractV2Tests(unittest.TestCase):

    def test_v1_and_v2_contracts_dispatch_without_mutating_v1(self) -> None:
        before_layered = copy.deepcopy(LAYERED_V1)
        before_funnel = copy.deepcopy(FUNNEL_V1)
        self.assertEqual(metrics.validate_contract(LAYERED_V1, "layered")["version"], 1)
        self.assertEqual(metrics.validate_contract(LAYERED_V2, "layered")["version"], 2)
        self.assertEqual(metrics.validate_contract(FUNNEL_V1, "funnel")["version"], 1)
        self.assertEqual(metrics.validate_contract(FUNNEL_V2, "funnel")["version"], 2)
        self.assertEqual(LAYERED_V1, before_layered)
        self.assertEqual(FUNNEL_V1, before_funnel)

    def test_unknown_or_crossed_versions_fail_closed(self) -> None:
        for contract, family in ((copy.deepcopy(LAYERED_V2), "layered"),
                                 (copy.deepcopy(FUNNEL_V2), "funnel")):
            contract["version"] = 99
            with self.assertRaises(metrics.ContractError):
                metrics.validate_contract(contract, family)
        crossed = copy.deepcopy(LAYERED_V2)
        crossed["schema"] = metrics.LAYERED_SCHEMAS[1]
        with self.assertRaises(metrics.ContractError):
            metrics.validate_contract(crossed, "layered")

    def test_v2_uses_lightweight_provenance(self) -> None:
        validity = LAYERED_V2["experiment_validity"]
        required = validity["required_identity"] + validity["required_population"]
        self.assertFalse(any("sha256" in name.lower() for name in required))
        policy = validity["provenance_policy"]
        self.assertIn("temporary_activation",
                      policy["sha256_not_required_for"])
        self.assertIn("runtime_overlay_file_tree",
                      policy["sha256_not_required_for"])
        self.assertIn("prefix_cache_content_identity",
                      policy["sha256_required_only_for"])

        funnel_evidence = [
            name
            for stage in FUNNEL_V2["stages"]
            for name in stage["required_evidence"]
        ]
        self.assertFalse(any("sha256" in name.lower()
                             for name in funnel_evidence))

    def test_report_binding_prevents_v1_report_from_becoming_v2(self) -> None:
        v1_report = {
            "schema": "bi100-validation-layer-report-v1",
            "version": 1,
            "status": "pass",
            "contract": {"schema": metrics.LAYERED_SCHEMAS[1], "version": 1},
        }
        self.assertEqual(metrics.validate_report_binding(
            v1_report, LAYERED_V1, "layered")["version"], 1)
        with self.assertRaises(metrics.ContractError):
            metrics.validate_report_binding(v1_report, LAYERED_V2, "layered")

    def test_four_v2_states_are_reachable_and_distinct(self) -> None:
        good = numeric()
        self.assertEqual(metrics.classify_fp16_numerics(
            good, LAYERED_V2)["status"], "pass")
        failed = copy.deepcopy(good)
        failed["candidate_relative_l2"] = 2.01e-4
        self.assertEqual(metrics.classify_fp16_numerics(
            failed, LAYERED_V2)["status"], "fail")
        drift = distribution()
        drift["high_margin_flips"] = 1
        self.assertEqual(metrics.classify_distribution(
            drift, {"shared_logprob_delta_p99": 0.01,
                    "paired_nll_upper_ci": 0.001}, LAYERED_V2)["status"],
                         "inconclusive")
        malformed = copy.deepcopy(good)
        del malformed["metadata_exact"]
        self.assertEqual(metrics.classify_fp16_numerics(
            malformed, LAYERED_V2)["status"], "invalid")

    def test_fp16_ratio_uses_denominator_floor_and_rounded_drift_is_diagnostic(self) -> None:
        value = numeric()
        value.update({"candidate_relative_l2": 2e-12,
                      "baseline_relative_l2": 0.0,
                      "candidate_max_abs": 2e-12,
                      "baseline_max_abs": 0.0})
        result = metrics.classify_fp16_numerics(value, LAYERED_V2)
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["relative_l2_error_ratio"], 2.0)
        self.assertEqual(result["maximum_absolute_error_ratio"], 2.0)

    def test_lse_limit_is_calibrated_to_baseline(self) -> None:
        value = numeric()
        value["baseline_lse_relative_l2"] = 1e-3
        value["candidate_lse_relative_l2"] = 2e-3
        self.assertEqual(metrics.classify_fp16_numerics(
            value, LAYERED_V2)["status"], "pass")

    def test_distribution_thresholds_come_from_aa_not_top1_floor(self) -> None:
        result = metrics.classify_distribution(
            distribution(), {"shared_logprob_delta_p99": 0.04,
                             "paired_nll_upper_ci": 0.006}, LAYERED_V2)
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["high_margin_threshold_nats"], 0.16)
        self.assertEqual(result["nll_regression_upper_ci_limit"], 0.012)
        self.assertEqual(result["top1_agreement"], 0.10)

    def test_capability_underpowered_is_inconclusive(self) -> None:
        evidence = {
            "deterministic_baseline_only_failures": 0,
            "paired_lower_ci": -0.01,
            "paired_bootstrap_reported": True,
            "exact_mcnemar_reported": True,
            "underpowered": True,
            "strata": {name: capability_stratum(underpowered=True)
                       for name in LAYERED_V2["paired_task_capability"]
                       ["required_strata"]},
        }
        self.assertEqual(metrics.classify_capability(
            evidence, LAYERED_V2, phase="development")["status"],
                         "inconclusive")

    def test_empty_capability_stratum_is_invalid(self) -> None:
        evidence = {
            "deterministic_baseline_only_failures": 0,
            "paired_lower_ci": 0.0,
            "paired_bootstrap_reported": True,
            "exact_mcnemar_reported": True,
            "underpowered": False,
            "strata": {name: capability_stratum() for name in
                       LAYERED_V2["paired_task_capability"]["required_strata"]},
        }
        evidence["strata"]["code"] = {}
        self.assertEqual(metrics.classify_capability(
            evidence, LAYERED_V2, phase="development")["status"], "invalid")

    def test_validity_rejects_empty_identity_and_nonfinite_timing(self) -> None:
        evidence = {
            name: f"identity-{name}"
            for name in LAYERED_V2["experiment_validity"]["required_identity"]
        }
        evidence.update({
            "expected_request_count": 2,
            "attempted_request_count": 2,
            "completed_request_count": 2,
            "failed_request_count": 0,
            "workload_order": "fixed",
            "workload_identity": "pair-1",
            "gpu_preflight": True,
            "gpu_postflight": True,
            "scoped_cleanup": True,
            "fatal_scan": True,
            "timing_samples": [1.0, 2.0],
        })
        self.assertEqual(metrics.classify_validity(
            evidence, LAYERED_V2)["status"], "pass")
        evidence["runtime_versions"] = ""
        self.assertEqual(metrics.classify_validity(
            evidence, LAYERED_V2)["status"], "invalid")
        evidence["runtime_versions"] = "corex-3.2.3"
        evidence["timing_samples"] = [float("nan")]
        self.assertEqual(metrics.classify_validity(
            evidence, LAYERED_V2)["status"], "invalid")
        evidence["timing_samples"] = [1.0]
        evidence["runtime_versions"] = True
        self.assertEqual(metrics.classify_validity(
            evidence, LAYERED_V2)["status"], "invalid")

    def test_performance_two_and_three_percent_boundaries(self) -> None:
        self.assertEqual(metrics.classify_performance(
            -0.05, -0.10, LAYERED_V2)["status"], "fail")
        self.assertEqual(metrics.classify_performance(
            0.0199, 0.01, LAYERED_V2)["status"], "fail")
        self.assertEqual(metrics.classify_performance(
            0.02, 0.01, LAYERED_V2)["status"], "inconclusive")
        self.assertEqual(metrics.classify_performance(
            0.0299, 0.01, LAYERED_V2)["status"], "inconclusive")
        self.assertEqual(metrics.classify_performance(
            0.03, 0.0001, LAYERED_V2)["status"], "pass")
        self.assertEqual(metrics.classify_performance(
            0.03, 0.0, LAYERED_V2)["status"], "inconclusive")

    def test_amdahl_projection(self) -> None:
        self.assertAlmostEqual(metrics.amdahl_projected_gain(0.2, 2.0),
                               1 / 0.9 - 1)


if __name__ == "__main__":
    unittest.main()
