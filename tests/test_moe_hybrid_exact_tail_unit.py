from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
SPEC = importlib.util.spec_from_file_location(
    "qualify_moe_hybrid_exact_tail",
    TESTS / "qualify_moe_hybrid_exact_tail.py",
)
QUALIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(QUALIFY)


def numerical(relative_l2: float = 0.0) -> dict:
    return {
        "exact": relative_l2 == 0,
        "finite": True,
        "max_abs": 0.0 if relative_l2 == 0 else 1.0e-5,
        "mean_abs": 0.0 if relative_l2 == 0 else 1.0e-6,
        "relative_l2": relative_l2,
    }


def report() -> dict:
    return {
        "schema": QUALIFY.BENCHMARK_SCHEMA,
        "version": 2,
        "shape": dict(QUALIFY.EXPECTED_SHAPE),
        "config": {"sequence_steps": 500},
        "extension_capabilities": {
            "w13": True,
            "w2_reduce": True,
            "w13_silu": False,
        },
        "numerics": {
            "direct_w13": numerical(5.0e-6),
            "hybrid_exact_tail": numerical(6.0e-6),
        },
        "sequence": {
            "hybrid_exact_tail": {
                "steps": 500,
                "exact_steps": 0,
                "finite_steps": 500,
                "relative_l2": 6.0e-6,
                "max_step_relative_l2": 9.0e-6,
            },
        },
        "timings": {
            "baseline_fixed": {"median_ms": 0.20},
            "hybrid_exact_tail_fixed": {"median_ms": 0.10},
            "baseline_routed": {"median_ms": 0.24},
            "hybrid_exact_tail_routed": {"median_ms": 0.12},
        },
    }


class MoeHybridExactTailTest(unittest.TestCase):

    def test_fixed_design_passes_all_component_gates(self):
        result = QUALIFY.qualify(report())
        self.assertTrue(result["component_qualified"], result)
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["observed"]["fixed_speedup"], 2.0)
        self.assertEqual(result["observed"]["routed_speedup"], 2.0)
        self.assertAlmostEqual(
            result["observed"]["routed_saving_ms"], 0.12)
        self.assertFalse(result["production_promotion_authorized"])

    def test_single_step_numerical_excess_fails_closed(self):
        value = report()
        value["sequence"]["hybrid_exact_tail"][
            "max_step_relative_l2"] = 1.01e-5
        result = QUALIFY.qualify(value)
        self.assertFalse(result["component_qualified"])
        self.assertTrue(any(
            "max_step_relative_l2" in reason
            for reason in result["reasons"]))

    def test_performance_shortfall_fails_without_relaxing_numerics(self):
        value = report()
        value["timings"]["hybrid_exact_tail_routed"][
            "median_ms"] = 0.20
        result = QUALIFY.qualify(value)
        self.assertFalse(result["component_qualified"])
        self.assertTrue(any(
            "routed speedup" in reason or "routed saving" in reason
            for reason in result["reasons"]))

    def test_nonfinite_or_wrong_shape_fails_closed(self):
        value = copy.deepcopy(report())
        value["shape"]["hidden"] = 1024
        value["numerics"]["hybrid_exact_tail"]["finite"] = False
        result = QUALIFY.qualify(value)
        self.assertFalse(result["component_qualified"])
        self.assertTrue(any(
            "target shape" in reason for reason in result["reasons"]))
        self.assertTrue(any(
            "non-finite" in reason for reason in result["reasons"]))

    def test_benchmark_uses_vendor_w2_and_exact_reduction(self):
        source = (
            TESTS / "bench_moe_direct_routed.py"
        ).read_text(encoding="utf-8")
        self.assertIn("hybrid_exact_tail_from_route", source)
        self.assertIn("torch.index_select(w2, 0, ids)", source)
        self.assertIn("torch.bmm(", source)
        self.assertIn("reducer.serial_float(expert_output, weights)", source)
        self.assertNotIn("hybrid_tile", source)
        self.assertNotIn("hybrid_threshold", source)


if __name__ == "__main__":
    unittest.main()
