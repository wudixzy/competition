from __future__ import annotations

import unittest

from tests.compare_fused_prefill_service_ab import compare


def _report(mode: str, cold_scale: float = 1.0,
            warm_scale: float = 1.0, output_tps: float = 22.0) -> dict:
    return {
        "schema": "bi100-m1-47-service-measurement-v1",
        "mode": mode,
        "run_id": "fixed",
        "max_tokens": 32,
        "qualified_measurement": True,
        "output_tps_p10": output_tps,
        "cases": [
            {
                "target_prompt_tokens": target,
                "cold": {"ttft_s": cold * cold_scale},
                "warm_ttft_median_s": warm * warm_scale,
            }
            for target, cold, warm in (
                (65536, 20.0, 0.5),
                (235000, 80.0, 0.7),
            )
        ],
    }


class FusedPrefillServiceComparisonUnitTest(unittest.TestCase):

    def test_accepts_candidate_at_all_fixed_boundaries(self):
        control = _report("control", output_tps=4.7)
        candidate = _report(
            "candidate", cold_scale=0.8, warm_scale=1.02,
            output_tps=4.606)
        result = compare(control, candidate)
        self.assertTrue(result["qualified"], result)
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(
            result["absolute_output_tps_gate"],
            "deferred_to_full_881_replay")

    def test_rejects_each_service_regression(self):
        control = _report("control")
        for candidate in (
                _report("candidate", cold_scale=0.81),
                _report("candidate", cold_scale=0.8, warm_scale=1.021),
                _report("candidate", cold_scale=0.8, output_tps=21.55)):
            with self.subTest(candidate=candidate):
                self.assertFalse(compare(control, candidate)["qualified"])

    def test_absolute_output_gate_is_reserved_for_full_replay(self):
        control = _report("control", output_tps=4.7)
        candidate = _report(
            "candidate", cold_scale=0.8, output_tps=4.606)
        result = compare(control, candidate)
        self.assertTrue(result["qualified"], result)
        self.assertEqual(
            result["thresholds"]["full_replay_minimum_output_tps_p10"],
            20.0)

    def test_rejects_unqualified_or_mismatched_measurements(self):
        control = _report("control")
        candidate = _report("candidate", cold_scale=0.8)
        candidate["run_id"] = "different"
        candidate["qualified_measurement"] = False
        result = compare(control, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn("run_id mismatch", result["reasons"])


if __name__ == "__main__":
    unittest.main()
