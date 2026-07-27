from __future__ import annotations

import copy
import unittest

from tests.qualify_gdn_combined_qk_norm import qualify


def valid_report() -> dict:
    exact = {
        "exact": True,
        "finite": True,
        "max_abs": 0.0,
        "relative_l2": 0.0,
    }
    return {
        "schema": "bi100-gdn-combined-qk-norm-v1",
        "config": {
            "fixed_seeds": [20260715, 20260727],
        },
        "fixed": {
            "20260715": {
                "normalized": copy.deepcopy(exact),
                "mapped": copy.deepcopy(exact),
            },
            "20260727": {
                "normalized": copy.deepcopy(exact),
                "mapped": copy.deepcopy(exact),
            },
        },
        "sequence": {
            "steps": 500,
            "finite_steps": 500,
            "normalized_exact_steps": 500,
            "mapped_exact_steps": 500,
            "normalized_relative_l2": 0.0,
            "mapped_relative_l2": 0.0,
            "max_normalized_relative_l2": 0.0,
            "max_mapped_relative_l2": 0.0,
        },
        "timings": {
            "candidate": {
                "speedup_vs_reference": 1.4,
                "saving_ms": 0.03,
                "projected_30_layer_saving_ms": 0.9,
            },
        },
    }


class CombinedQkNormQualificationTests(unittest.TestCase):
    def test_valid_component_passes_without_authorizing_production(self):
        result = qualify(valid_report())
        self.assertTrue(result["component_qualified"])
        self.assertFalse(result["production_promotion_authorized"])
        self.assertEqual(result["reasons"], [])

    def test_nonexact_sequence_fails(self):
        report = valid_report()
        report["sequence"]["mapped_exact_steps"] = 499
        result = qualify(report)
        self.assertFalse(result["component_qualified"])
        self.assertIn(
            "sequence mapped_exact_steps is not exact", result["reasons"])

    def test_relative_l2_boundary_is_inclusive(self):
        report = valid_report()
        report["sequence"]["mapped_relative_l2"] = 1.0e-5
        self.assertTrue(qualify(report)["component_qualified"])
        report["sequence"]["mapped_relative_l2"] = 1.0001e-5
        self.assertFalse(qualify(report)["component_qualified"])

    def test_speed_and_saving_are_independent_gates(self):
        report = valid_report()
        report["timings"]["candidate"]["speedup_vs_reference"] = 1.24
        report["timings"]["candidate"]["saving_ms"] = 0.019
        reasons = qualify(report)["reasons"]
        self.assertIn("q/k stage speedup is below 1.25x", reasons)
        self.assertIn(
            "q/k stage saving is below 0.02 ms/layer", reasons)


if __name__ == "__main__":
    unittest.main()
