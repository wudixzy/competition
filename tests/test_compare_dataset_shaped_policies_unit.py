import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/compare_dataset_shaped_policies.py"
SPEC = importlib.util.spec_from_file_location("policy_compare", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def summary(output: float, hit: float, score: float, ttft: float = 4.0):
    return {
        "validation": {
            "complete_matrix": True,
            "success_rate": 1.0,
        },
        "aggregate": {
            "output_tps_p10": output,
            "input_tps_aggregate": 700.0,
            "cache_tps_aggregate": 7000.0,
            "ttft_p90_all_s": ttft,
            "cache_hit_rate": hit,
            "weighted_score": score,
        },
    }


class DatasetPolicyCompareTest(unittest.TestCase):

    def test_candidate_passes_stage_gates_at_boundaries(self):
        report = MODULE.compare(
            summary(21.0, 0.50, 6000.0),
            summary(20.58, 0.55, 6300.0),
        )
        self.assertTrue(report["stage_qualified"])
        self.assertIsNone(report["capacity_256k_preserved"])
        self.assertIsNone(report["final_qualified"])

    def test_candidate_fails_on_output_regression(self):
        report = MODULE.compare(
            summary(21.0, 0.50, 6000.0),
            summary(20.0, 0.56, 6500.0),
        )
        self.assertFalse(report["stage_qualified"])
        self.assertFalse(
            report["stage_gates"]["output_tps_regression_at_most_2pct"])

    def test_final_metrics_do_not_imply_capacity_qualification(self):
        report = MODULE.compare(
            summary(21.0, 0.50, 6000.0),
            summary(22.0, 0.60, 8100.0),
        )
        self.assertTrue(report["final_metric_gates_passed"])
        self.assertIsNone(report["final_qualified"])


if __name__ == "__main__":
    unittest.main()
