from __future__ import annotations

import copy
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "compare_m1_99_fused_prefill_paired_ab.py"
SPEC = importlib.util.spec_from_file_location("m1_99_comparator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def digest(value: int) -> str:
    return f"{value:064x}"


def request(
    output: int,
    first: int = 1,
    *,
    prompt_tokens: int,
    cached_tokens: int,
) -> dict:
    return {
        "ttft_s": 1.0,
        "output_sha256": digest(output),
        "first_token_sha256": digest(first),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": 32,
        "finish_reason": "length",
    }


def report(
    mode: str,
    run_id: str,
    *,
    short_cold: float,
    long_cold: float,
    output_tps: float = 21.0,
) -> dict:
    cases = []
    for target, output, cold_ttft in (
        (32768, 5, short_cold),
        (65536, 10, short_cold),
        (131072, 15, short_cold),
        (235000, 20, long_cold),
    ):
        cold = request(
            output,
            prompt_tokens=target,
            cached_tokens=0,
        )
        cold["ttft_s"] = cold_ttft
        warm_1 = request(
            output,
            prompt_tokens=target,
            cached_tokens=target - 16,
        )
        warm_2 = request(
            output,
            prompt_tokens=target,
            cached_tokens=target - 16,
        )
        cases.append({
            "target_prompt_tokens": target,
            "cold": cold,
            "warm_1": warm_1,
            "warm_2": warm_2,
            "warm_ttft_median_s": 1.0,
        })
    return {
        "schema": MODULE.MEASUREMENT_SCHEMA,
        "mode": mode,
        "run_id": run_id,
        "max_tokens": 32,
        "qualified_measurement": True,
        "reasons": [],
        "output_tps_p10": output_tps,
        "cases": cases,
    }


def valid_pairs() -> tuple[list[dict], list[dict]]:
    controls = []
    candidates = []
    for index in range(3):
        run_id = f"pair-{index + 1}"
        controls.append(report(
            "control", run_id, short_cold=100.0, long_cold=500.0))
        candidates.append(report(
            "candidate", run_id, short_cold=99.0, long_cold=465.0))
    return controls, candidates


class M199FusedPrefillComparatorUnitTest(unittest.TestCase):

    def test_valid_three_pair_screen_qualifies(self):
        controls, candidates = valid_pairs()
        result = MODULE.compare(controls, candidates)
        self.assertTrue(result["qualified"], result["reasons"])
        self.assertTrue(result["quality"]["required_output_gate_passed"])
        self.assertTrue(
            result["decision"]["full_tp4_quality_gate_authorized"])
        self.assertFalse(result["decision"]["production_promotion_authorized"])

    def test_235k_full_output_divergence_is_rejected(self):
        controls, candidates = valid_pairs()
        for candidate in candidates:
            for case in candidate["cases"]:
                if case["target_prompt_tokens"] == 235000:
                    case["cold"]["output_sha256"] = digest(999)
        result = MODULE.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertFalse(
            result["quality"]["all_full_output_hashes_exact"])
        self.assertEqual(
            len(result["quality"]["long_full_output_mismatches"]), 3)
        self.assertTrue(result["quality"]["long_full_output_exact_required"])

    def test_first_token_divergence_is_always_rejected(self):
        controls, candidates = valid_pairs()
        candidates[0]["cases"][-1]["cold"]["first_token_sha256"] = digest(999)
        result = MODULE.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "pair[1] target 235000 cold first generated token differs",
            result["reasons"],
        )

    def test_65k_full_output_divergence_is_rejected(self):
        controls, candidates = valid_pairs()
        candidates[0]["cases"][1]["warm_1"]["output_sha256"] = digest(999)
        result = MODULE.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "pair[1] target 65536 warm_1 full output differs",
            result["reasons"],
        )

    def test_old_small_long_context_gain_still_fails_revised_screen(self):
        controls, candidates = valid_pairs()
        for candidate in candidates:
            candidate["cases"][-1]["cold"]["ttft_s"] = 480.0
        result = MODULE.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "235K median cold improvement" in reason
            for reason in result["reasons"]
        ))

    def test_single_arm_outlier_regression_is_rejected(self):
        controls, candidates = valid_pairs()
        candidates[0]["output_tps_p10"] = 18.0
        result = MODULE.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "Output TPS regression" in reason
            for reason in result["reasons"]
        ))

    def test_warm_fallback_uses_absolute_jitter_screen(self):
        controls, candidates = valid_pairs()
        for candidate in candidates:
            for case in candidate["cases"]:
                case["warm_ttft_median_s"] = 1.20
        result = MODULE.compare(controls, candidates)
        self.assertTrue(result["qualified"], result["reasons"])
        self.assertAlmostEqual(
            result["target_summary"]["235000"][
                "warm_slowdown_median_s"],
            0.20,
        )

    def test_warm_residual_mismatch_is_rejected(self):
        controls, candidates = valid_pairs()
        candidates[0]["cases"][-1]["warm_1"]["cached_tokens"] -= 1
        result = MODULE.compare(controls, candidates)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "warm residual differs" in reason
            for reason in result["reasons"]
        ))

    def test_inputs_are_not_mutated(self):
        controls, candidates = valid_pairs()
        original = copy.deepcopy((controls, candidates))
        MODULE.compare(controls, candidates)
        self.assertEqual((controls, candidates), original)

    def test_exactly_three_pairs_are_required_without_crashing(self):
        controls, candidates = valid_pairs()
        result = MODULE.compare(controls[:2], candidates[:2])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["quality"]["required_output_gate_passed"])
        self.assertIn(
            "exactly 3 control/candidate pairs required",
            result["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
