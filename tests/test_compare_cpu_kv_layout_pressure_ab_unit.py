from __future__ import annotations

import pathlib
import sys
import unittest


TESTS = pathlib.Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import compare_cpu_kv_layout_pressure_ab as comparator


NAMES = (
    "target_cold",
    "target_immediate_warm",
    "pressure_cold_0000",
    "target_after_pressure",
    "target_refreshed",
)


def _report(times: dict[str, float]) -> dict:
    requests = []
    for name in NAMES:
        requests.append({
            "expected_prompt_tokens": 65_536,
            "name": name,
            "status": "ok",
            "summary": {
                "cached_tokens": 0 if name == "target_cold" else 65_520,
                "completion_tokens": 8,
                "elapsed_s": times.get(name, 1.0),
                "finish_reason": "length",
                "message_sha256": "a" * 64,
                "prompt_tokens": 65_536,
            },
        })
    return {
        "params": {
            "json_out": "ignored.json",
            "mode": "candidate",
            "pressure_count": 1,
            "run_id": "fixed",
        },
        "qualified": True,
        "requests": requests,
        "schema": comparator.INPUT_SCHEMA,
        "validation": {"qualified": True, "reasons": []},
        "version": 1,
    }


class LayoutPressureComparatorTest(unittest.TestCase):

    def test_fixed_gate_qualifies(self):
        paged = _report({
            "target_after_pressure": 10.0,
            "target_immediate_warm": 2.0,
            "target_refreshed": 2.0,
        })
        candidate = _report({
            "target_after_pressure": 7.9,
            "target_immediate_warm": 2.03,
            "target_refreshed": 2.04,
        })
        report = comparator.compare(paged, candidate)
        self.assertTrue(report["qualified"], report)

    def test_after_pressure_failure_is_closed(self):
        paged = _report({"target_after_pressure": 10.0})
        candidate = _report({"target_after_pressure": 8.01})
        report = comparator.compare(paged, candidate)
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("target_after_pressure" in reason
                            for reason in report["reasons"]))

    def test_warm_regression_failure_is_closed(self):
        paged = _report({"target_after_pressure": 10.0})
        candidate = _report({
            "target_after_pressure": 7.0,
            "target_immediate_warm": 1.021,
        })
        report = comparator.compare(paged, candidate)
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("target_immediate_warm" in reason
                            for reason in report["reasons"]))

    def test_response_mismatch_is_rejected(self):
        paged = _report({"target_after_pressure": 10.0})
        candidate = _report({"target_after_pressure": 7.0})
        candidate["requests"][3]["summary"]["message_sha256"] = "b" * 64
        report = comparator.compare(paged, candidate)
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("message_sha256" in reason
                            for reason in report["reasons"]))

    def test_both_inputs_must_pass_retention(self):
        paged = _report({"target_after_pressure": 10.0})
        candidate = _report({"target_after_pressure": 7.0})
        paged["qualified"] = False
        report = comparator.compare(paged, candidate)
        self.assertFalse(report["qualified"], report)
        self.assertIn("paged retention gate is not qualified", report["reasons"])


if __name__ == "__main__":
    unittest.main()
