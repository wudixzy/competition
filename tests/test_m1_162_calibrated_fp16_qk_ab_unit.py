from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(TESTS))
SCRIPT = TESTS / "bench_m1_162_calibrated_fp16_qk_ab.py"
SPEC = importlib.util.spec_from_file_location("m1_162_cell", SCRIPT)
CELL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CELL)


def calibrated(relative_multiple=1.1, max_multiple=1.2):
    return {
        "candidate_finite": True,
        "reference_fp32_finite": True,
        "rounded_reference_finite": True,
        "candidate_vs_rounded_relative_l2": 2.0e-5,
        "candidate_vs_rounded_max_abs": 6.0e-5,
        "candidate_to_fp32_relative_l2": 2.2e-4,
        "fp16_rounding_to_fp32_relative_l2": 2.0e-4,
        "relative_l2_error_multiple_over_fp16_rounding": relative_multiple,
        "candidate_to_fp32_max_abs": 1.2e-3,
        "fp16_rounding_to_fp32_max_abs": 1.0e-3,
        "max_abs_error_multiple_over_fp16_rounding": max_multiple,
    }


def valid_result(case="p90_total_32k_q8176"):
    context_len, query_len = CELL.CASES[case]
    trials = [100.0, 99.0, 101.0, 98.0, 102.0]
    return {
        "schema": CELL.SCHEMA,
        "case": case,
        "context_len": context_len,
        "query_len": query_len,
        "warmups": CELL.WARMUPS,
        "trials": CELL.TRIALS,
        "seed": CELL.CASE_SEEDS[case],
        "numeric_contract_sha256": CELL.CONTRACT_SHA256,
        "numerical": {
            "candidate_calibrated": calibrated(),
            "baseline_calibrated": calibrated(1.0, 1.0),
            "candidate_lse_relative_l2": 3.0e-8,
            "candidate_repeat": {
                "output_exact": True,
                "lse_exact": True,
            },
        },
        "timings": {
            "baseline": {"cuda_trials_ms": trials},
            "candidate": {
                "cuda_trials_ms": [value / 1.15 for value in trials],
            },
            "speedup": 1.15,
        },
        "authorization": {
            "operator_screen_only": True,
            "real_activation_replay_authorized": False,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }


class M1162CalibratedFp16QkTest(unittest.TestCase):
    def test_contract_assigns_rounded_difference_a_diagnostic_role(self):
        contract = json.loads(
            (ROOT / "quality" / "fused_prefill_numeric_adjudication.v2.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(
            contract["diagnostic_roles"][
                "candidate_vs_rounded_relative_l2"],
            "distribution_drift_characterization",
        )
        self.assertFalse(
            contract["hard_gates"]["semantic_evidence_may_waive_failure"])
        self.assertNotIn(
            "maximum_candidate_vs_rounded_relative_l2",
            contract["hard_gates"],
        )

    def test_candidate_above_legacy_l2_can_pass_calibrated_numeric_gate(self):
        result = valid_result()
        self.assertGreater(
            result["numerical"]["candidate_calibrated"][
                "candidate_vs_rounded_relative_l2"],
            1.0e-5,
        )
        evaluation = CELL.evaluate(result)
        self.assertTrue(evaluation["qualified"], evaluation["reasons"])

    def test_rounding_multiple_nonfinite_and_repeat_fail_closed(self):
        result = valid_result()
        result["numerical"]["candidate_calibrated"][
            "relative_l2_error_multiple_over_fp16_rounding"
        ] = 2.01
        self.assertFalse(CELL.evaluate(result)["qualified"])

        result = valid_result()
        result["numerical"]["candidate_calibrated"][
            "candidate_finite"
        ] = False
        self.assertFalse(CELL.evaluate(result)["qualified"])

        result = valid_result()
        result["numerical"]["candidate_repeat"]["output_exact"] = False
        self.assertFalse(CELL.evaluate(result)["qualified"])

    def test_contract_identity_and_authorization_are_fail_closed(self):
        result = copy.deepcopy(valid_result())
        result["numeric_contract_sha256"] = "0" * 64
        result["authorization"]["tp4_service_authorized"] = True
        evaluation = CELL.evaluate(result)
        self.assertFalse(evaluation["qualified"])
        self.assertGreaterEqual(len(evaluation["reasons"]), 2)

    def test_fresh_grid_uses_distinct_fixed_seeds(self):
        self.assertEqual(len(set(CELL.CASE_SEEDS.values())), 3)
        self.assertEqual(min(CELL.CASE_SEEDS.values()), 20260730)

    def test_wrapper_changes_only_experiment_identity_and_next_stage(self):
        source = (
            ROOT / "scripts" / "run_m1_162_calibrated_fp16_qk_ab.py"
        ).read_text(encoding="utf-8")
        self.assertIn("bench_m1_162_calibrated_fp16_qk_ab.py", source)
        self.assertIn(
            '"real_activation_replay_authorized": qualified', source)
        self.assertIn('"short_tp4_screen_authorized": False', source)


if __name__ == "__main__":
    unittest.main()
