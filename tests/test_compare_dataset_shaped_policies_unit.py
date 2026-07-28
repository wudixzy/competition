import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/compare_dataset_shaped_policies.py"
SPEC = importlib.util.spec_from_file_location("policy_compare", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def summary(output: float, hit: float, score: float, ttft: float = 4.0):
    requests = []
    for target in (4096, 7800, 16000):
        for pair in (1, 2, 3):
            for phase in ("cold", "warm"):
                requests.append({
                    "path": f"{target}_pair{pair}_{phase}.json",
                    "target": target,
                    "pair": pair,
                    "phase": phase,
                    "prompt_salt": f"fixed_{target}_{pair}",
                    "rendered_tokens_local": target,
                    "cached_tokens": (
                        0 if target == 4096 and pair == 1
                        and phase == "cold"
                        else target // (2 if phase == "warm" else 4)
                    ),
                })
    return {
        "validation": {
            "complete_matrix": True,
            "success_rate": 1.0,
            "token_count_match": True,
            "target_within_one_block": True,
            "cold_warm_pair_salts_match": True,
        },
        "aggregate": {
            "output_tps_p10": output,
            "input_tps_aggregate": 700.0,
            "cache_tps_aggregate": 7000.0,
            "ttft_p90_all_s": ttft,
            "cache_hit_rate": hit,
            "weighted_score": score,
        },
        "requests": requests,
    }


class DatasetPolicyCompareTest(unittest.TestCase):

    def test_historical_m1_35_clears_both_v2_benefit_paths(self):
        report = MODULE.compare(
            summary(21.6563, 0.499301, 6699.4888, ttft=20.8748),
            summary(21.7783, 0.610671, 6976.7204, ttft=18.0882),
        )
        self.assertTrue(report["stage_qualified"])
        self.assertTrue(report["benefit_paths"][
            "effective_hit_gain_at_least_2pp"])
        self.assertTrue(report["benefit_paths"][
            "weighted_score_gain_at_least_3pct_without_hit_reduction"])
        self.assertAlmostEqual(
            report["delta"]["effective_hit_percentage_points"],
            11.137,
        )
        self.assertAlmostEqual(
            report["delta"]["weighted_score_fraction"],
            0.041381,
            places=5,
        )
        self.assertIsNone(report["quality_nonregression_qualified"])
        self.assertIsNone(report["final_qualified"])

    def test_candidate_passes_stage_gates_at_boundaries(self):
        report = MODULE.compare(
            summary(21.0, 0.50, 6000.0),
            summary(20.58, 0.52, 6000.0, ttft=4.08),
        )
        self.assertTrue(report["stage_qualified"])
        self.assertTrue(report["benefit_paths"][
            "effective_hit_gain_at_least_2pp"])
        self.assertFalse(report["benefit_paths"][
            "weighted_score_gain_at_least_3pct_without_hit_reduction"])
        self.assertIsNone(report["quality_nonregression_qualified"])
        self.assertIsNone(report["capacity_256k_preserved"])
        self.assertIsNone(report["final_qualified"])

    def test_weighted_path_passes_without_hit_reduction(self):
        report = MODULE.compare(
            summary(21.0, 0.55, 6000.0),
            summary(20.58, 0.55, 6180.0, ttft=4.08),
        )
        self.assertTrue(report["stage_qualified"])
        self.assertFalse(report["benefit_paths"][
            "effective_hit_gain_at_least_2pp"])
        self.assertTrue(report["benefit_paths"][
            "weighted_score_gain_at_least_3pct_without_hit_reduction"])

    def test_weighted_path_rejects_hit_reduction(self):
        report = MODULE.compare(
            summary(21.0, 0.55, 6000.0),
            summary(21.0, 0.54, 6300.0),
        )
        self.assertFalse(report["stage_qualified"])
        self.assertFalse(report["stage_gates"][
            "cache_benefit_path_qualified"])

    def test_candidate_fails_without_either_benefit_path(self):
        report = MODULE.compare(
            summary(21.0, 0.55, 6000.0),
            summary(21.0, 0.56, 6179.0),
        )
        self.assertFalse(report["stage_qualified"])
        self.assertFalse(report["stage_gates"][
            "cache_benefit_path_qualified"])

    def test_candidate_fails_on_output_regression(self):
        report = MODULE.compare(
            summary(21.0, 0.50, 6000.0),
            summary(20.0, 0.56, 6500.0),
        )
        self.assertFalse(report["stage_qualified"])
        self.assertFalse(
            report["stage_gates"]["output_tps_regression_at_most_2pct"])

    def test_candidate_fails_on_ttft_regression(self):
        report = MODULE.compare(
            summary(21.0, 0.50, 6000.0, ttft=4.0),
            summary(21.0, 0.53, 6200.0, ttft=4.081),
        )
        self.assertFalse(report["stage_qualified"])
        self.assertFalse(
            report["stage_gates"]["ttft_p90_regression_at_most_2pct"])

    def test_candidate_must_reach_final_hit_floor(self):
        report = MODULE.compare(
            summary(21.0, 0.46, 6000.0),
            summary(21.0, 0.49, 6200.0),
        )
        self.assertFalse(report["stage_qualified"])
        self.assertFalse(report["stage_gates"][
            "effective_cache_hit_at_least_50pct"])

    def test_final_metrics_do_not_imply_capacity_qualification(self):
        report = MODULE.compare(
            summary(21.0, 0.50, 6000.0),
            summary(22.0, 0.60, 8100.0),
        )
        self.assertTrue(report["final_metric_gates_passed"])
        self.assertIsNone(report["final_qualified"])

    def test_different_prompt_contract_rejects_candidate(self):
        baseline = summary(21.0, 0.50, 6000.0)
        candidate = summary(22.0, 0.60, 7000.0)
        candidate["requests"][0]["prompt_salt"] = "different"
        report = MODULE.compare(baseline, candidate)
        self.assertFalse(report["stage_qualified"])
        self.assertFalse(
            report["stage_gates"]["request_contract_identical"])

    def test_invalid_baseline_rejects_comparison(self):
        baseline = summary(21.0, 0.50, 6000.0)
        baseline["validation"]["success_rate"] = 0.98
        report = MODULE.compare(
            baseline,
            summary(21.0, 0.53, 6200.0),
        )
        self.assertFalse(report["stage_qualified"])
        self.assertFalse(report["stage_gates"][
            "baseline_success_rate_at_least_99pct"])

    def test_first_request_must_be_uncached(self):
        baseline = summary(21.0, 0.50, 6000.0)
        candidate = summary(21.0, 0.53, 6200.0)
        baseline["requests"][0]["cached_tokens"] = 16
        candidate["requests"][0]["cached_tokens"] = 16
        report = MODULE.compare(baseline, candidate)
        self.assertFalse(report["stage_qualified"])
        self.assertFalse(
            report["stage_gates"]["baseline_first_request_uncached"])
        self.assertFalse(report["stage_gates"]["first_request_uncached"])

    def test_warm_cache_must_not_fall_below_cold(self):
        baseline = summary(21.0, 0.50, 6000.0)
        candidate = summary(21.0, 0.53, 6200.0)
        candidate["requests"][3]["cached_tokens"] = 0
        report = MODULE.compare(baseline, candidate)
        self.assertFalse(report["stage_qualified"])
        self.assertFalse(report["stage_gates"]["pair_cache_monotonic"])


if __name__ == "__main__":
    unittest.main()
