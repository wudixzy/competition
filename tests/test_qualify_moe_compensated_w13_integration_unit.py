from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
QUALIFIER_PATH = (
    ROOT / "tests" / "qualify_moe_compensated_w13_integration.py")
SPEC = importlib.util.spec_from_file_location(
    "qualify_moe_compensated_w13_integration",
    QUALIFIER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
QUALIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFIER)

SHA_A = "a" * 64
SHA_B = "b" * 64


def comparison(
    *,
    relative_l2: float,
    max_abs: float,
    mismatches: int,
) -> dict[str, object]:
    return {
        "exact": mismatches == 0,
        "finite": True,
        "mismatch_count": mismatches,
        "max_abs": max_abs,
        "mean_abs": max_abs / 4.0,
        "relative_l2": relative_l2,
    }


def sequence(
    *,
    relative_l2: float,
    max_step_relative_l2: float,
    max_abs: float,
    mismatches: int,
) -> dict[str, object]:
    return {
        "steps": 500,
        "rows": 500 * 2048,
        "finite_steps": 500,
        "exact_steps": 500 if mismatches == 0 else 0,
        "mismatch_count": mismatches,
        "max_abs": max_abs,
        "mean_abs": max_abs / 4.0,
        "relative_l2": relative_l2,
        "max_step_relative_l2": max_step_relative_l2,
    }


def timing(value: float) -> dict[str, object]:
    trials = [value] * 9
    return {
        "median_ms": value,
        "p10_ms": value,
        "p90_ms": value,
        "trials_ms": trials,
    }


def valid_report() -> dict[str, object]:
    fixed = {}
    sequences = {}
    for seed in (20260716, 20260727):
        fixed[str(seed)] = {
            "direct_vs_reference": comparison(
                relative_l2=2.0e-5,
                max_abs=9.765625e-4,
                mismatches=1200,
            ),
            "candidate_vs_reference": comparison(
                relative_l2=1.5e-5,
                max_abs=4.8828125e-4,
                mismatches=900,
            ),
            "candidate_vs_direct": comparison(
                relative_l2=1.0e-5,
                max_abs=4.8828125e-4,
                mismatches=700,
            ),
            "direct_repeat_exact": True,
            "candidate_repeat_exact": True,
        }
        sequences[str(seed)] = {
            "direct_vs_reference": sequence(
                relative_l2=2.0e-5,
                max_step_relative_l2=4.0e-5,
                max_abs=1.953125e-3,
                mismatches=50000,
            ),
            "candidate_vs_reference": sequence(
                relative_l2=1.8e-5,
                max_step_relative_l2=3.5e-5,
                max_abs=9.765625e-4,
                mismatches=45000,
            ),
        }
    orders = [
        list(QUALIFIER.TIMING_CASES) if repeat % 2 == 0
        else list(reversed(QUALIFIER.TIMING_CASES))
        for repeat in range(9)
    ]
    return {
        "schema": QUALIFIER.SCHEMA,
        "version": 1,
        "device": "BI-V100",
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
            "w13_weight_scale": 0.02,
            "w2_weight_scale": 0.02,
        },
        "method": {
            "reference":
                "pytorch_gather_linear_vllm_silu_and_mul_bmm_"
                "corex_serial_float_reduce",
            "control": "production_direct_w13_and_w2_reduce",
            "candidate":
                "production_compensated_w13_and_same_w2_reduce",
            "timing_order": "alternating_forward_reverse",
            "request_semantics_changed": False,
        },
        "artifacts": {
            "extension_sha256": SHA_A,
            "extension_size_bytes": 1000,
            "exact_reduce_sha256": SHA_B,
            "exact_reduce_size_bytes": 1000,
        },
        "fixed": fixed,
        "sequence": sequences,
        "timings": {
            "cases": {
                "strict_reference": timing(2.0),
                "direct_control": timing(1.0),
                "compensated_candidate": timing(1.01),
            },
            "orders": orders,
            "candidate_vs_direct_ratio": 1.01,
            "candidate_vs_reference_speedup": 2.0 / 1.01,
        },
    }


class CompensatedW13IntegrationQualificationUnitTest(unittest.TestCase):

    def qualify(self, report: dict[str, object]) -> dict[str, object]:
        return QUALIFIER.qualify(
            report,
            expected_extension_sha256=SHA_A,
            expected_exact_reduce_sha256=SHA_B,
            report_sha256="c" * 64,
        )

    def test_fixed_contract_qualifies_without_authorizing_promotion(
        self,
    ) -> None:
        result = self.qualify(valid_report())
        self.assertTrue(result["qualified"], result["reasons"])
        self.assertTrue(
            result["decision"]["tp4_evaluation_authorized"])
        self.assertFalse(
            result["decision"]["production_promotion_authorized"])
        self.assertFalse(result["decision"]["yaml_change_authorized"])
        self.assertFalse(result["decision"]["main_merge_authorized"])

    def test_candidate_numerical_regression_is_rejected(self) -> None:
        report = valid_report()
        candidate = report["sequence"]["20260716"][
            "candidate_vs_reference"
        ]
        candidate["relative_l2"] = 3.0e-5
        result = self.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "relative_l2" in reason for reason in result["reasons"]))

    def test_candidate_routed_regression_over_two_percent_is_rejected(
        self,
    ) -> None:
        report = valid_report()
        report["timings"]["cases"]["compensated_candidate"] = timing(1.03)
        report["timings"]["candidate_vs_direct_ratio"] = 1.03
        report["timings"]["candidate_vs_reference_speedup"] = 2.0 / 1.03
        result = self.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "more than 2%" in reason for reason in result["reasons"]))

    def test_nonfinite_or_tampered_artifact_fails_closed(self) -> None:
        report = valid_report()
        report["fixed"]["20260727"][
            "candidate_vs_reference"
        ]["finite"] = False
        result = self.qualify(report)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "invalid benchmark evidence",
            result["reasons"][0],
        )

        tampered = copy.deepcopy(valid_report())
        tampered["artifacts"]["extension_sha256"] = "d" * 64
        result = self.qualify(tampered)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "SHA-256" in reason for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
