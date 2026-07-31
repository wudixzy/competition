from __future__ import annotations

import copy
import unittest

from tests import summarize_m1_170_order_balance as module


def comparison(overhead: float) -> dict:
    return {
        "schema": module.COMPARISON_SCHEMA,
        "version": 2,
        "qualified_analysis": True,
        "reasons": [],
        "request_manifest_sha256": "a" * 64,
        "cold": {
            "request_count": 9,
            "admission64_overhead_fraction_median": overhead,
            "admission64_overhead_fraction_p90": overhead / 2,
        },
        "cold_isolation": {
            "qualified": True,
            "admission64_cold_cached_tokens": 0,
            "off_cold_cached_tokens": 0,
        },
        "by_shape": {
            str(shape): {"admission64_overhead_fraction": overhead}
            for shape in (4096, 7800, 16000)
        },
        "cross_policy_numeric_observation": {
            "first_token_identity_rate": 1.0,
            "complete_output_identity_rate": 8 / 9,
            "cold_first_token_identity_rate": 1.0,
            "cold_complete_output_identity_rate": 8 / 9,
        },
    }


def runner(order: list[str]) -> dict:
    return {
        "schema": module.RUNNER_SCHEMA,
        "version": 2,
        "source_revision": "b" * 40,
        "qualified_development_screen": True,
        "returncode": 0,
        "arm_order": order,
        "bench_tool_count": 0,
        "gates": {"comparison": 0, "cleanup": 0},
    }


class M1170OrderBalanceUnitTest(unittest.TestCase):

    def test_order_balanced_geometric_overhead(self) -> None:
        report = module.summarize(
            comparison(0.10), comparison(-0.10),
            runner(["admission64", "off"]),
            runner(["off", "admission64"]),
        )
        self.assertTrue(report["qualified_order_balanced_timing"])
        self.assertAlmostEqual(
            report["median"][
                "order_balanced_geometric_overhead_fraction"],
            (0.99 ** 0.5) - 1.0,
        )
        self.assertFalse(
            report["scope"]["statistical_significance_claimed"])
        self.assertFalse(
            report["scope"]["production_promotion_authorized"])

    def test_cold_cache_contamination_is_rejected(self) -> None:
        forward = comparison(0.1)
        forward["cold_isolation"][
            "admission64_cold_cached_tokens"] = 16
        with self.assertRaisesRegex(ValueError, "qualified cold screen"):
            module.summarize(
                forward, comparison(0.0),
                runner(["admission64", "off"]),
                runner(["off", "admission64"]),
            )

    def test_runner_order_and_manifest_are_bound(self) -> None:
        with self.assertRaisesRegex(ValueError, "runner contract differs"):
            module.summarize(
                comparison(0.1), comparison(0.0),
                runner(["off", "admission64"]),
                runner(["off", "admission64"]),
            )
        reverse = copy.deepcopy(comparison(0.0))
        reverse["request_manifest_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "manifests differ"):
            module.summarize(
                comparison(0.1), reverse,
                runner(["admission64", "off"]),
                runner(["off", "admission64"]),
            )


if __name__ == "__main__":
    unittest.main()
