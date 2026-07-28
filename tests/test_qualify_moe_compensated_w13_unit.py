from __future__ import annotations

import copy
import unittest
from pathlib import Path

from tests.qualify_moe_compensated_w13 import qualify


CANDIDATE_SHA256 = "a" * 64
DIRECT_SHA256 = "b" * 64


EXACT_SEQUENCE_INDICES = [0, 1, 2, 3, 7, 15, 31, 63, 127, 255, 383, 499]


def comparison(
    relative_l2: float,
    *,
    mismatches: int = 1,
    max_abs: float = 0.001,
) -> dict[str, object]:
    return {
        "exact": mismatches == 0,
        "finite": True,
        "mismatch_count": mismatches,
        "max_abs": max_abs,
        "mean_abs": min(max_abs, 0.000001),
        "relative_l2": relative_l2,
    }


def sequence(
    relative_l2: float,
    max_step: float,
    *,
    steps: int = 500,
    mismatches: int | None = None,
    max_abs: float = 0.001,
) -> dict[str, object]:
    if mismatches is None:
        mismatches = steps
    return {
        "steps": steps,
        "rows": steps * 2048,
        "finite_steps": steps,
        "exact_steps": 0,
        "mismatch_count": mismatches,
        "max_abs": max_abs,
        "mean_abs": min(max_abs, 0.000001),
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
    exact_sequences = {}
    for seed in (20260716, 20260727):
        fixed[str(seed)] = {
            "vendor_vs_exact": comparison(
                5.0e-6,
                mismatches=10,
                max_abs=0.002,
            ),
            "direct_vs_exact": comparison(
                8.0e-6,
                mismatches=15,
                max_abs=0.003,
            ),
            "direct_vs_vendor": comparison(2.0e-5),
            "compensated_vs_vendor": comparison(1.5e-5),
            "compensated_vs_exact": comparison(
                4.0e-6,
                mismatches=8,
                max_abs=0.001,
            ),
        }
        sequences[str(seed)] = {
            "direct": sequence(1.5e-5, 2.0e-5),
            "compensated": sequence(1.4e-5, 6.0e-5),
        }
        exact_sequences[str(seed)] = {
            "sample_indices": EXACT_SEQUENCE_INDICES,
            "comparisons": {
                "vendor": sequence(
                    7.0e-6,
                    1.0e-5,
                    steps=len(EXACT_SEQUENCE_INDICES),
                    mismatches=120,
                    max_abs=0.003,
                ),
                "direct": sequence(
                    9.0e-6,
                    1.2e-5,
                    steps=len(EXACT_SEQUENCE_INDICES),
                    mismatches=150,
                    max_abs=0.004,
                ),
                "compensated": sequence(
                    5.0e-6,
                    8.0e-6,
                    steps=len(EXACT_SEQUENCE_INDICES),
                    mismatches=100,
                    max_abs=0.002,
                ),
            },
        }
    return {
        "schema": "bi100-moe-compensated-w13-v2",
        "version": 2,
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
            "exact_sequence_indices": EXACT_SEQUENCE_INDICES,
        },
        "method": {
            "algorithm": "per_lane_kahan_fp32_then_rn_warp_tree",
            "quality_reference":
                "cpu_float64_dot_rounded_to_fp16_noninferiority",
            "exact_diagnostic":
                "fixed_fixture_and_stratified_sequence_samples",
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
        "exact_sequence": exact_sequences,
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
        self.assertTrue(
            result["decision"]["single_gpu_numerical_screen_qualified"])
        self.assertFalse(
            result["decision"]["production_promotion_authorized"])
        self.assertFalse(result["decision"]["yaml_change_authorized"])
        self.assertFalse(result["decision"]["main_merge_authorized"])

    def test_exact_sequence_noninferiority_regression_fails(self) -> None:
        report = valid_report()
        report["exact_sequence"]["20260727"]["comparisons"][
            "compensated"]["relative_l2"] = 8.0e-6
        result = self.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "candidate relative L2 is worse than vendor" in reason
            for reason in result["reasons"]
        ))

    def test_candidate_vendor_gap_is_diagnostic_only(self) -> None:
        report = valid_report()
        for seed in ("20260716", "20260727"):
            report["fixed"][seed]["compensated_vs_vendor"][
                "relative_l2"] = 3.0e-5
            report["sequence"][seed]["compensated"][
                "relative_l2"] = 3.0e-5
            report["sequence"][seed]["compensated"][
                "max_step_relative_l2"] = 8.0e-5
        result = self.qualify(report)
        self.assertTrue(result["qualified"], result["reasons"])

    def test_fixed_exact_reference_regression_fails(self) -> None:
        report = valid_report()
        report["fixed"]["20260716"]["compensated_vs_exact"][
            "mismatch_count"] = 11
        result = self.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "candidate mismatch count is worse than vendor" in reason
            for reason in result["reasons"]
        ))

    def test_exact_sample_selection_is_fixed(self) -> None:
        report = valid_report()
        report["exact_sequence"]["20260716"]["sample_indices"] = [0, 1]
        result = self.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(result["reasons"][0].startswith(
            "invalid benchmark evidence:"))

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
