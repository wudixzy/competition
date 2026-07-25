import copy
import importlib.util
import math
import statistics
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "bench_m1_55_production_prefill.py"
SPEC = importlib.util.spec_from_file_location("m1_55_bench", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_result(case_name="production_65k_q8176"):
    context_len, query_len, kind = MODULE.CASES[case_name]
    reference_trials = [100.0, 101.0, 102.0]
    candidate_trials = [50.0, 51.0, 52.0]
    result = {
        "schema": MODULE.SCHEMA,
        "case": case_name,
        "context_len": context_len,
        "query_len": query_len,
        "kind": kind,
        "seed": MODULE.SEED,
        "warmups": MODULE.WARMUPS,
        "trials": MODULE.TRIALS,
        "physical_block_permutation": context_len > 0,
        "numerical": {
            "finite": True,
            "output_relative_l2": 5e-6,
            "lse_relative_l2": 1e-6,
            "output_max_abs": 6e-5,
        },
        "timings": {
            "reference": {
                "cuda_trials_ms": reference_trials,
                "cuda_median_ms": statistics.median(reference_trials),
            },
            "candidate": {
                "cuda_trials_ms": candidate_trials,
                "cuda_median_ms": statistics.median(candidate_trials),
            },
            "speedup": (
                statistics.median(reference_trials)
                / statistics.median(candidate_trials)
            ),
        },
        "authorization": {
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }
    return result


class M155ProductionPrefillUnitTest(unittest.TestCase):
    def test_valid_production_cell_qualifies_component_gate(self):
        evaluation = MODULE.evaluate_cell(valid_result())
        self.assertTrue(evaluation["qualified"], evaluation["reasons"])

    def test_real_production_shapes_are_frozen(self):
        self.assertEqual(
            MODULE.CASES["production_65k_q8176"],
            (65_536, 8_176, "production"),
        )
        self.assertEqual(
            MODULE.CASES["production_128k_q8176"],
            (122_880, 8_176, "production"),
        )
        self.assertEqual(
            MODULE.CASES["production_235k_q5616"],
            (229_376, 5_616, "production"),
        )

    def test_small_and_paged_numerical_cases_are_frozen(self):
        self.assertEqual(
            MODULE.CASES["golden_dense_q1"], (0, 1, "numerical"))
        self.assertEqual(
            MODULE.CASES["golden_paged_240_q16"],
            (240, 16, "numerical"),
        )
        self.assertEqual(
            MODULE.CASES["boundary_234992_q8"],
            (234_992, 8, "numerical"),
        )

    def test_dense_case_has_no_synthetic_block_permutation(self):
        result = valid_result("production_dense_q8176")
        evaluation = MODULE.evaluate_cell(result)
        self.assertTrue(evaluation["qualified"], evaluation["reasons"])

    def test_q256_cannot_qualify_production_shape(self):
        result = valid_result("legacy_74k_q256")
        result["timings"]["speedup"] = 1.1
        evaluation = MODULE.evaluate_cell(result)
        self.assertTrue(evaluation["qualified"], evaluation["reasons"])
        self.assertEqual(result["kind"], "legacy")

    def test_production_speedup_below_gate_is_rejected(self):
        result = valid_result()
        result["timings"]["speedup"] = 1.49
        evaluation = MODULE.evaluate_cell(result)
        self.assertFalse(evaluation["qualified"])
        self.assertTrue(any(
            "below 1.5x" in reason for reason in evaluation["reasons"]))

    def test_numerical_failure_is_rejected(self):
        result = valid_result()
        result["numerical"]["output_relative_l2"] = 1.1e-5
        evaluation = MODULE.evaluate_cell(result)
        self.assertFalse(evaluation["qualified"])
        self.assertTrue(any(
            "output_relative_l2" in reason
            for reason in evaluation["reasons"]))

    def test_nonfinite_is_rejected(self):
        result = valid_result()
        result["numerical"]["finite"] = False
        result["numerical"]["lse_relative_l2"] = math.inf
        evaluation = MODULE.evaluate_cell(result)
        self.assertFalse(evaluation["qualified"])

    def test_timing_median_must_match_trials(self):
        result = valid_result()
        result["timings"]["candidate"]["cuda_median_ms"] = 1.0
        evaluation = MODULE.evaluate_cell(result)
        self.assertFalse(evaluation["qualified"])
        self.assertTrue(any(
            "does not match trials" in reason
            for reason in evaluation["reasons"]))

    def test_authorization_fails_closed(self):
        result = valid_result()
        result["authorization"]["main_or_yaml_change_authorized"] = True
        evaluation = MODULE.evaluate_cell(result)
        self.assertFalse(evaluation["qualified"])

    def test_split4_workspace_exposes_full_query_growth(self):
        q256 = MODULE.split4_aux_workspace_bytes(256)
        q8176 = MODULE.split4_aux_workspace_bytes(8_176)
        q5616 = MODULE.split4_aux_workspace_bytes(5_616)
        self.assertGreater(q8176, 400 * 1024 * 1024)
        self.assertGreater(q5616, 250 * 1024 * 1024)
        self.assertGreater(q8176 / q256, 20)

    def test_mutated_case_shape_is_rejected(self):
        result = copy.deepcopy(valid_result())
        result["query_len"] = 256
        evaluation = MODULE.evaluate_cell(result)
        self.assertFalse(evaluation["qualified"])


if __name__ == "__main__":
    unittest.main()
