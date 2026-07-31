from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import unittest

from tests import bench_m1_104_admission64_policy_matrix as matrix
from tests import compare_m1_170_cold_capture_overhead as compare


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_m1_170_cold_capture_overhead_tp1_ab.sh"
RUNNER = ROOT / "scripts" / "run_m1_169_tail64_nofinal_tp1_ab.sh"


def _row(target: int, pair: int, phase: str, ttft: float) -> dict:
    identity = f"{target}-{pair}-{phase}"
    return {
        "request_id": identity,
        "target_prompt_tokens": target,
        "pair": pair,
        "phase": phase,
        "salt_sha256": "a" * 64,
        "rendered_tokens_local": target,
        "seed": matrix.SEED,
        "ttft_s": ttft,
        "cached_tokens": 0,
        "first_token_sha256": identity,
        "output_sha256": identity,
        "content_sha256": identity,
        "reasoning_sha256": identity,
        "tool_calls_sha256": identity,
        "finish_reason": "stop",
        "completion_tokens": 8,
    }


def _report(policy: str, ttft: float) -> dict:
    rows = [
        _row(target, pair, phase, ttft)
        for target in matrix.SHAPES
        for pair in matrix.PAIRS
        for phase in matrix.PHASES
    ]
    return {
        "schema": matrix.SCHEMA,
        "version": matrix.VERSION,
        "policy": policy,
        "qualified_measurement": True,
        "reasons": [],
        "request_count": matrix.REQUEST_COUNT,
        "request_manifest_sha256": "b" * 64,
        "fixed": {"salt_order": "identity-first", "tool_count": 0},
        "requests": rows,
    }


class M1170ColdCaptureOverheadUnitTest(unittest.TestCase):

    def test_wrapper_is_bounded_diagnostic(self) -> None:
        subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("CANDIDATE_POLICY=off", source)
        self.assertIn("BENCH_SALT_ORDER=identity-first", source)
        self.assertIn("BENCH_TOOL_COUNT=0", source)
        self.assertIn("exec", source)

    def test_shared_runner_keeps_m1_169_default(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("CANDIDATE_POLICY=${CANDIDATE_POLICY:-tail64_nofinal}",
                      source)
        self.assertIn("compare_m1_170_cold_capture_overhead.py", source)
        self.assertIn('"production_promotion_authorized": False', source)

    def test_comparison_attributes_admission_overhead(self) -> None:
        control = _report("admission64", 1.2)
        candidate = _report("off", 1.0)
        result = compare.compare(control, candidate)
        self.assertTrue(result["qualified_analysis"])
        self.assertAlmostEqual(
            result["cold"]["admission64_overhead_fraction_median"], 0.2)
        self.assertEqual(
            result["cross_policy_numeric_observation"][
                "complete_output_identity_rate"], 1.0)
        self.assertFalse(
            result["scope"]["production_promotion_authorized"])

    def test_output_difference_is_separate_from_timing_analysis(self) -> None:
        control = _report("admission64", 1.0)
        candidate = _report("off", 1.0)
        candidate = copy.deepcopy(candidate)
        candidate["requests"][0]["output_sha256"] = "different"
        result = compare.compare(control, candidate)
        self.assertTrue(result["qualified_analysis"])
        self.assertEqual(
            result["cross_policy_numeric_observation"][
                "complete_output_identity_matches"],
            matrix.REQUEST_COUNT - 1,
        )
        self.assertFalse(result["cross_policy_numeric_observation"][
            "strict_output_identity_qualified"])
        self.assertFalse(result["cross_policy_numeric_observation"][
            "teacher_forced_logits_evaluated"])

    def test_cache_off_must_report_zero_cached_tokens(self) -> None:
        control = _report("admission64", 1.0)
        candidate = _report("off", 1.0)
        candidate = copy.deepcopy(candidate)
        candidate["requests"][0]["cached_tokens"] = 16
        result = compare.compare(control, candidate)
        self.assertFalse(result["qualified_analysis"])
        self.assertIn(
            "cold rows contain cached tokens and cannot attribute capture "
            "overhead", result["reasons"])

    def test_admission64_cold_rows_must_also_be_uncached(self) -> None:
        control = _report("admission64", 1.0)
        candidate = _report("off", 1.0)
        control = copy.deepcopy(control)
        control["requests"][0]["cached_tokens"] = 16
        result = compare.compare(control, candidate)
        self.assertFalse(result["qualified_analysis"])
        self.assertEqual(
            result["cold_isolation"]["admission64_cold_cached_tokens"], 16)


if __name__ == "__main__":
    unittest.main()
