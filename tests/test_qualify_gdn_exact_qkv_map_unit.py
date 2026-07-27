from __future__ import annotations

import copy
import unittest

from tests.qualify_gdn_exact_qkv_map import qualify


def valid_report() -> dict:
    exact = {
        "exact": True,
        "finite": True,
        "max_abs": 0.0,
        "relative_l2": 0.0,
    }
    return {
        "schema": "bi100-gdn-exact-qkv-map-v1",
        "shape": {
            "batch": 1,
            "key_heads": 4,
            "value_heads": 8,
            "head_dim": 128,
            "dtype": "torch.float16",
        },
        "artifact": {
            "path": "/tmp/corex_gdn_qkv_map.so",
            "sha256": "a" * 64,
        },
        "config": {
            "fixed_seeds": [20260715, 20260727],
            "sequence_steps": 1000,
        },
        "fixed": {
            str(seed): {
                "mapped_qk": copy.deepcopy(exact),
                "value": copy.deepcopy(exact),
                "complete": copy.deepcopy(exact),
            }
            for seed in (20260715, 20260727)
        },
        "sequence": {
            "steps": 1000,
            "exact_steps": 1000,
            "finite_steps": 1000,
            "relative_l2": 0.0,
            "max_relative_l2": 0.0,
            "max_abs": 0.0,
        },
        "timings": {
            "pair": {
                "reference": {"median_ms": 0.08},
                "candidate": {"median_ms": 0.05},
                "paired_speedup_median": 1.6,
            },
            "candidate_speedup": 1.6,
            "candidate_saving_ms": 0.03,
            "projected_30_layer_saving_ms": 0.9,
        },
        "production_integration_attempted": False,
    }


class ExactQkvMapQualificationTests(unittest.TestCase):

    def test_valid_component_does_not_authorize_production(self):
        result = qualify(valid_report())
        self.assertTrue(result["component_qualified"])
        self.assertFalse(result["production_promotion_authorized"])
        self.assertEqual(result["reasons"], [])

    def test_any_nonexact_value_fails(self):
        report = valid_report()
        report["fixed"]["20260715"]["value"]["exact"] = False
        report["sequence"]["exact_steps"] = 999
        reasons = qualify(report)["reasons"]
        self.assertIn(
            "fixed seed 20260715 value is not exact", reasons)
        self.assertIn("sequence output is not exact", reasons)

    def test_numerical_limit_is_inclusive(self):
        report = valid_report()
        report["sequence"]["max_relative_l2"] = 1.0e-5
        self.assertTrue(qualify(report)["component_qualified"])
        report["sequence"]["max_relative_l2"] = 1.0001e-5
        self.assertFalse(qualify(report)["component_qualified"])

    def test_speed_and_saving_are_independent_gates(self):
        report = valid_report()
        report["timings"]["candidate_speedup"] = 1.24
        report["timings"]["pair"]["paired_speedup_median"] = 1.23
        report["timings"]["candidate_saving_ms"] = 0.019
        reasons = qualify(report)["reasons"]
        self.assertIn("q/k/v stage speedup is below 1.25x", reasons)
        self.assertIn("paired q/k/v speedup is below 1.25x", reasons)
        self.assertIn(
            "q/k/v stage saving is below 0.02 ms/layer", reasons)

    def test_wrong_shape_and_missing_artifact_fail(self):
        report = valid_report()
        report["shape"]["value_heads"] = 4
        report["artifact"]["sha256"] = ""
        reasons = qualify(report)["reasons"]
        self.assertIn(
            "benchmark shape differs from the TP4 production shape", reasons)
        self.assertIn("extension artifact SHA-256 is invalid", reasons)


if __name__ == "__main__":
    unittest.main()
