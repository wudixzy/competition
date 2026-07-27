#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import qualify_moe_exact_w2_hybrid as qualification  # noqa: E402


def passing_report() -> dict:
    return {
        "schema": "bi100-moe-exact-w2-hybrid-v1",
        "shape": dict(qualification.EXPECTED_SHAPE),
        "checks": {
            "selected_w2_exact": True,
            "direct_w13": {"finite": True, "relative_l2": 7.0e-6},
            "hybrid": {"finite": True, "relative_l2": 8.0e-6},
        },
        "sequence": {
            "steps": 500,
            "finite_steps": 500,
            "relative_l2": 8.5e-6,
            "max_step_relative_l2": 9.5e-6,
        },
        "timings": {
            "hybrid_fixed": {"speedup_vs_baseline": 1.6},
            "hybrid_routed": {"speedup_vs_baseline": 1.3},
        },
    }


class MoeExactW2HybridQualificationUnitTest(unittest.TestCase):
    def test_accepts_candidate_at_all_hard_gates(self) -> None:
        result = qualification.qualify(passing_report())
        self.assertTrue(result["qualified"])
        self.assertEqual(result["reasons"], [])

    def test_rejects_non_exact_w2_and_numerical_drift(self) -> None:
        report = passing_report()
        report["checks"]["selected_w2_exact"] = False
        report["checks"]["hybrid"]["relative_l2"] = 1.1e-5
        report["sequence"]["max_step_relative_l2"] = 1.2e-5
        result = qualification.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("selected W2" in row for row in result["reasons"]))
        self.assertTrue(any("hybrid relative" in row
                            for row in result["reasons"]))
        self.assertTrue(any("max_step" in row for row in result["reasons"]))

    def test_rejects_short_sequence_and_weak_speedup(self) -> None:
        report = copy.deepcopy(passing_report())
        report["sequence"]["steps"] = 499
        report["sequence"]["finite_steps"] = 499
        report["timings"]["hybrid_fixed"]["speedup_vs_baseline"] = 1.49
        report["timings"]["hybrid_routed"]["speedup_vs_baseline"] = 1.24
        result = qualification.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("at least 500" in row
                            for row in result["reasons"]))
        self.assertTrue(any("fixed speedup" in row
                            for row in result["reasons"]))
        self.assertTrue(any("routed speedup" in row
                            for row in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
