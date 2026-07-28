from __future__ import annotations

import copy
import unittest
from pathlib import Path

from tests.qualify_moe_compensated_w13 import qualify


CANDIDATE_SHA256 = "a" * 64
DIRECT_SHA256 = "b" * 64


def comparison(relative_l2: float) -> dict[str, object]:
    return {
        "exact": False,
        "finite": True,
        "mismatch_count": 1,
        "max_abs": 0.001,
        "mean_abs": 0.000001,
        "relative_l2": relative_l2,
    }


def sequence(relative_l2: float, max_step: float) -> dict[str, object]:
    return {
        "steps": 500,
        "rows": 500 * 2048,
        "finite_steps": 500,
        "exact_steps": 0,
        "mismatch_count": 500,
        "max_abs": 0.001,
        "mean_abs": 0.000001,
        "relative_l2": relative_l2,
        "max_step_relative_l2": max_step,
    }


def timing(milliseconds: float) -> dict[str, object]:
    trials = [milliseconds] * 9
    return {
        "median_ms": milliseconds,
        "p10_ms": milliseconds,
        "p90_ms": milliseconds,
        "trials_ms": trials,
    }


def valid_report() -> dict[str, object]:
    fixed = {}
    sequences = {}
    for seed in (20260716, 20260727):
        fixed[str(seed)] = {
            "vendor_vs_exact": comparison(3.0e-6),
            "direct_vs_vendor": comparison(2.0e-5),
            "compensated_vs_vendor": comparison(5.0e-6),
            "compensated_vs_exact": comparison(4.0e-6),
        }
        sequences[str(seed)] = {
            "direct": sequence(1.5e-5, 2.0e-5),
            "compensated": sequence(5.0e-6, 8.0e-6),
        }
    return {
        "schema": "bi100-moe-compensated-w13-v1",
        "version": 1,
        "device": "Iluvatar BI-V100",
        "shape": {
            "experts": 256,
            "top_k": 8,
            "hidden": 2048,
            "intermediate": 128,
            "rows_per_expert": 256,
            "dtype": "torch.float16",
        },
        "config": {
            "device": "cuda:0",
            "seeds": [20260716, 20260727],
            "sequence_steps_per_seed": 500,
            "warmup": 30,
            "iterations": 300,
            "repeats": 9,
            "cpu_threads": 8,
            "weight_scale": 0.02,
        },
        "method": {
            "algorithm": "per_lane_kahan_fp32_then_rn_warp_tree",
            "quality_reference": "torch_nn_functional_linear_fp16",
            "exact_diagnostic":
                "cpu_float64_dot_rounded_to_fp16_fixed_fixture_only",
            "fixture_generation":
                "hidden_then_router_then_w13_then_sequence",
            "production_runtime_changed": False,
        },
        "extensions": {
            "candidate_sha256": CANDIDATE_SHA256,
            "candidate_size_bytes": 100,
            "direct_sha256": DIRECT_SHA256,
            "direct_size_bytes": 200,
        },
        "fixed": fixed,
        "sequence": sequences,
        "timings": {
            "cases": {
                "vendor_fixed": timing(3.0),
                "direct_fixed": timing(0.8),
                "compensated_fixed": timing(1.0),
                "vendor_routed": timing(3.0),
                "direct_routed": timing(1.5),
                "compensated_routed": timing(2.0),
            },
            "speedups": {
                "compensated_fixed_vs_vendor": 3.0,
                "compensated_routed_vs_vendor": 1.5,
            },
        },
    }


class CompensatedW13QualificationTests(unittest.TestCase):
    def qualify(self, report: dict[str, object]) -> dict[str, object]:
        return qualify(
            report,
            expected_candidate_sha256=CANDIDATE_SHA256,
            expected_direct_sha256=DIRECT_SHA256,
            report_sha256="c" * 64,
        )

    def test_valid_report_qualifies_without_promotion(self) -> None:
        result = self.qualify(valid_report())
        self.assertTrue(result["qualified"], result["reasons"])
        self.assertTrue(result["decision"]["single_gpu_probe_qualified"])
        self.assertFalse(
            result["decision"]["production_promotion_authorized"])
        self.assertFalse(result["decision"]["yaml_change_authorized"])
        self.assertFalse(result["decision"]["main_merge_authorized"])

    def test_candidate_numerical_regression_fails(self) -> None:
        report = valid_report()
        report["sequence"]["20260727"]["compensated"][
            "max_step_relative_l2"] = 1.1e-5
        result = self.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "max-step relative L2 exceeds" in reason
            for reason in result["reasons"]
        ))

    def test_fixture_must_reproduce_direct_gap(self) -> None:
        report = valid_report()
        for seed in ("20260716", "20260727"):
            report["fixed"][seed]["direct_vs_vendor"][
                "relative_l2"] = 5.0e-6
            report["sequence"][seed]["direct"]["relative_l2"] = 5.0e-6
            report["sequence"][seed]["direct"][
                "max_step_relative_l2"] = 8.0e-6
        result = self.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "fixture did not reproduce the production direct W13 gap",
            result["reasons"],
        )

    def test_extension_hash_mismatch_fails(self) -> None:
        report = valid_report()
        report["extensions"]["candidate_sha256"] = "d" * 64
        result = self.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "candidate extension SHA-256 does not match artifact",
            result["reasons"],
        )

    def test_inconsistent_timing_evidence_fails_closed(self) -> None:
        report = valid_report()
        report["timings"]["cases"]["compensated_fixed"]["median_ms"] = 0.5
        result = self.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(result["reasons"][0].startswith(
            "invalid benchmark evidence:"))

    def test_extra_field_fails_closed(self) -> None:
        report = valid_report()
        report["unexpected"] = True
        result = self.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(result["reasons"][0].startswith(
            "invalid benchmark evidence:"))


class CompensatedW13StaticContractTests(unittest.TestCase):
    def test_kernel_uses_fixed_shape_and_compensated_rounding(self) -> None:
        source = Path(
            "tests/corex_moe_compensated_w13_ext.cu"
        ).read_text(encoding="utf-8")
        for fragment in (
            "constexpr int kExperts = 256;",
            "constexpr int kTopK = 8;",
            "constexpr int kHidden = 2048;",
            "constexpr int kRowsPerExpert = 256;",
            "__fmul_rn",
            "__fadd_rn",
            "subtract_rn",
            "__float2half_rn",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("fmaf(", source)
        self.assertNotIn("__fsub_rn", source)

    def test_probe_is_not_wired_into_production_or_yaml(self) -> None:
        candidate_name = "corex_moe_compensated_w13"
        self.assertNotIn(
            candidate_name,
            Path("computility-run.yaml").read_text(encoding="utf-8"),
        )
        production_files = (
            Path("qwen3_6_scripts/qwen3_5.py"),
            Path("qwen3_6_scripts/corex_moe_direct_routed.cu"),
        )
        for path in production_files:
            self.assertNotIn(
                candidate_name,
                path.read_text(encoding="utf-8"),
            )

    def test_build_script_targets_only_the_probe(self) -> None:
        script = Path(
            "tests/build_corex_moe_compensated_w13.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "corex_moe_compensated_w13_ext.cu",
            script,
        )
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_moe_compensated_w13",
            script,
        )
        self.assertNotIn("qwen3_6_scripts", script)


if __name__ == "__main__":
    unittest.main()
