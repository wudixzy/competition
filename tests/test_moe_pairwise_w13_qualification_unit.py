#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import unittest


TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import qualify_moe_pairwise_w13 as qualification  # noqa: E402


def passing_report() -> dict:
    return {
        "schema": "bi100-moe-pairwise-w13-v1",
        "config": {"fixed_seeds": list(qualification.EXPECTED_SEEDS)},
        "fixed": {
            str(seed): {
                "pairwise": {"finite": True, "relative_l2": 8.0e-6},
            }
            for seed in qualification.EXPECTED_SEEDS
        },
        "sequence": {
            "pairwise": {
                "steps": 500,
                "finite_steps": 500,
                "relative_l2": 8.5e-6,
                "max_step_relative_l2": 9.5e-6,
            },
        },
        "timings": {
            "pairwise_fixed": {"speedup_vs_reference": 1.6},
            "pairwise_routed": {"speedup_vs_reference": 1.3},
        },
    }


class MoePairwiseW13QualificationUnitTest(unittest.TestCase):
    def test_accepts_candidate_at_all_hard_gates(self) -> None:
        result = qualification.qualify(passing_report())
        self.assertTrue(result["qualified"])
        self.assertEqual(result["reasons"], [])

    def test_rejects_one_bad_seed_and_sequence_tail(self) -> None:
        report = passing_report()
        report["fixed"]["20260727"]["pairwise"]["relative_l2"] = 1.1e-5
        report["sequence"]["pairwise"]["max_step_relative_l2"] = 1.2e-5
        result = qualification.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("20260727" in row for row in result["reasons"]))
        self.assertTrue(any("max_step" in row for row in result["reasons"]))

    def test_rejects_weak_speedup_and_short_sequence(self) -> None:
        report = passing_report()
        report["sequence"]["pairwise"]["steps"] = 499
        report["sequence"]["pairwise"]["finite_steps"] = 499
        report["timings"]["pairwise_fixed"]["speedup_vs_reference"] = 1.49
        report["timings"]["pairwise_routed"]["speedup_vs_reference"] = 1.24
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
