from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.qualify_moe_w13_rounding_guard import qualify


ROOT = Path(__file__).resolve().parents[1]
QUALIFIER = ROOT / "tests" / "qualify_moe_w13_rounding_guard.py"


def comparison(relative_l2: float = 0.0, mismatches: int = 0) -> dict:
    return {
        "exact": mismatches == 0,
        "mismatch_count": mismatches,
        "max_abs": 0.0 if mismatches == 0 else 0.0001,
        "mean_abs": 0.0 if mismatches == 0 else 1.0e-7,
        "relative_l2": relative_l2,
        "finite": True,
    }


def fixed_record() -> dict:
    return {
        "rows": 2048,
        "finite": True,
        "production_forward_exact": True,
        "flags": 8,
        "flagged_fraction": 8 / 2048,
        "vendor_mismatches": 4,
        "flagged_vendor_mismatches": 4,
        "missed_vendor_mismatches": 0,
        "false_positive_flags": 4,
        "mismatch_recall": 1.0,
        "flag_precision": 0.5,
        "exact_flag_mismatches": 0,
        "direct": comparison(2.0e-5, 4),
        "reverse": comparison(2.5e-5, 6),
        "exact_half": comparison(),
        "corrected": comparison(),
    }


def sequence_record() -> dict:
    return {
        "steps": 500,
        "rows": 500 * 2048,
        "finite_steps": 500,
        "flags": 3500,
        "flagged_fraction": 3500 / (500 * 2048),
        "max_step_flagged_fraction": 0.02,
        "vendor_mismatches": 2000,
        "flagged_vendor_mismatches": 2000,
        "missed_vendor_mismatches": 0,
        "false_positive_flags": 1500,
        "mismatch_recall": 1.0,
        "flag_precision": 2000 / 3500,
        "exact_flag_mismatches": 0,
        "direct_rms_step_relative_l2": 2.2e-5,
        "exact_rms_step_relative_l2": 0.0,
        "corrected_rms_step_relative_l2": 0.0,
        "max_exact_step_relative_l2": 0.0,
        "max_corrected_step_relative_l2": 0.0,
    }


def valid_report() -> dict:
    return {
        "schema": "bi100-moe-w13-rounding-guard-v1",
        "version": 1,
        "device": "Iluvatar BI-V100",
        "shape": {
            "experts": 256,
            "top_k": 8,
            "hidden": 2048,
            "intermediate": 128,
            "dtype": "torch.float16",
        },
        "config": {
            "device": "cuda:0",
            "seeds": [20260716, 20260727],
            "sequence_steps_per_seed": 500,
            "cpu_threads": 8,
        },
        "method": {
            "flag_rule": "forward_fp16_differs_from_reverse_fp16",
            "correction_oracle":
                "float64_dot_rounded_to_fp16_for_flagged_rows",
            "fixture_generation":
                "hidden_then_router_then_w13_then_sequence",
            "production_runtime_changed": False,
        },
        "fixtures": [
            {
                "seed": seed,
                "fixed": fixed_record(),
                "sequence": sequence_record(),
                "elapsed_s": 20.0,
            }
            for seed in (20260716, 20260727)
        ],
    }


class MoeW13RoundingGuardQualificationTest(unittest.TestCase):
    def test_valid_report_authorizes_only_next_bounded_kernel(self):
        result = qualify(valid_report())
        self.assertTrue(result["qualified"])
        self.assertTrue(
            result["decision"]["bounded_correction_kernel_authorized"])
        self.assertFalse(
            result["decision"]["production_promotion_authorized"])
        self.assertFalse(result["decision"]["yaml_change_authorized"])
        self.assertFalse(result["decision"]["main_merge_authorized"])

    def test_missed_vendor_mismatch_fails(self):
        report = valid_report()
        report["fixtures"][0]["sequence"]["missed_vendor_mismatches"] = 1
        result = qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "misses a sequence vendor mismatch" in reason
            for reason in result["reasons"]))

    def test_large_flag_set_fails(self):
        report = valid_report()
        report["fixtures"][1]["fixed"]["flagged_fraction"] = 0.051
        result = qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "fixed flagged fraction exceeds 5%" in reason
            for reason in result["reasons"]))

    def test_numerical_limit_fails_closed(self):
        report = valid_report()
        report["fixtures"][0]["sequence"][
            "max_corrected_step_relative_l2"] = 1.01e-5
        result = qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "corrected output exceeds relative L2 limit" in reason
            for reason in result["reasons"]))

    def test_inconsistent_counters_fail_closed(self):
        report = valid_report()
        report["fixtures"][0]["sequence"]["flags"] += 1
        result = qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "sequence mismatch counters are inconsistent" in reason
            or "sequence flagged fraction is inconsistent" in reason
            for reason in result["reasons"]))

    def test_non_exact_correction_fails_closed(self):
        report = valid_report()
        report["fixtures"][0]["fixed"]["corrected"] = comparison(
            relative_l2=1.0e-7,
            mismatches=1,
        )
        result = qualify(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "corrected output is not vendor-exact" in reason
            for reason in result["reasons"]))

    def test_logical_device_is_bound(self):
        report = valid_report()
        report["config"]["device"] = "cuda:1"
        result = qualify(report)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "single-GPU logical device contract changed",
            result["reasons"],
        )

    def test_fixture_order_is_bound(self):
        report = valid_report()
        report["fixtures"].reverse()
        result = qualify(report)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "fixture order or seed identity changed", result["reasons"])

    def test_probe_must_reproduce_existing_gap(self):
        report = valid_report()
        for fixture in report["fixtures"]:
            fixture["fixed"]["vendor_mismatches"] = 0
            fixture["sequence"]["vendor_mismatches"] = 0
        result = qualify(report)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "probe did not reproduce the production W13 numerical gap",
            result["reasons"],
        )

    def test_cli_returns_nonzero_for_rejected_report(self):
        report = valid_report()
        report["fixtures"][0]["fixed"]["exact_flag_mismatches"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            output_path = root / "qualification.json"
            report_path.write_text(
                json.dumps(report) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(QUALIFIER),
                    "--report",
                    str(report_path),
                    "--out",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertFalse(
                json.loads(output_path.read_text(
                    encoding="utf-8"))["qualified"])

    def test_input_is_not_mutated(self):
        report = valid_report()
        before = copy.deepcopy(report)
        qualify(report)
        self.assertEqual(report, before)


if __name__ == "__main__":
    unittest.main()
