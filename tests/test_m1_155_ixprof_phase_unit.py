from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import profile_m1_155_fused_prefill_phase as workload
import qualify_m1_155_ixprof_phase as qualification

RUNNER = (
    ROOT / "scripts" / "run_m1_155_fused_prefill_phase_profile.py"
).read_text(encoding="ascii")


def cell() -> dict:
    context_len, query_len = workload.CASES["p90_total_16k_q8176"]
    value = {
        "schema": workload.SCHEMA,
        "version": 1,
        "source_revision": "a" * 40,
        "instance": "test",
        "visible_physical_gpu": 1,
        "case": "p90_total_16k_q8176",
        "context_len": context_len,
        "query_len": query_len,
        "profile_trials": workload.PROFILE_TRIALS,
        "profile_cuda_ms": 100.0,
        "profiler_start_rc": 0,
        "profiler_stop_rc": 0,
        "expected_launches": workload.expected_launches(
            context_len, query_len),
        "extension": {"sha256": "b" * 64},
        "numerical": {
            "finite": True,
            "output_relative_l2": 6e-6,
            "lse_relative_l2": 2e-8,
            "output_max_abs": 2e-4,
        },
        "authorization": {
            "implementation_direction_authorized": False,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }
    value["evaluation"] = workload.evaluate(value)
    return value


def profile_log(value: dict) -> list[str]:
    expected = value["expected_launches"]
    rows = [
        ("convert_query", expected["convert_query"],
         "(anonymous namespace)::convert_query_kernel(__half const*)"),
        ("gather", expected["gather"],
         "(anonymous namespace)::gather_kv_group_kernel(__half const*)"),
        ("qk", expected["qk"],
         "void Gemm_tcu_bi_kernel::gemm_stride_tcuh2<float, true, false>"),
        ("mask", expected["mask"],
         "(anonymous namespace)::mask_group_scores_kernel(float*)"),
        ("normalize", expected["normalize"],
         "(anonymous namespace)::normalize_split_scores_kernel(float*)"),
        ("pv", expected["pv"],
         "void Gemm_tcu_bi_kernel::gemm_stride_tcuh2<float, false, false>"),
        ("merge", expected["merge"],
         "(anonymous namespace)::merge_split_output_kernel(float*)"),
    ]
    lines = []
    for index, (_, calls, name) in enumerate(rows):
        prefix = " GPU activities:" if index == 0 else "                "
        lines.append(
            f"{prefix}  10.00%  10.000ms  {calls}  1.0us  1.0us  1.0us  "
            f"{name}"
        )
    lines.append(
        "                 30.00%  30.000ms  3  1.0us  1.0us  1.0us  "
        "void at::native::modern::elementwise_kernel<DivFunctor<float>>"
    )
    lines.append("      API Calls: 100.00%  1.000ms  1  1us  1us  1us  x")
    return lines


class M1155IxprofPhaseUnitTest(unittest.TestCase):

    def test_runner_profiles_one_fixed_case_per_gpu_with_scoped_cleanup(self):
        self.assertIn('"--profile-from-start", "off"', RUNNER)
        self.assertIn("cleanup_children(", RUNNER)
        self.assertIn("_run_postflight(", RUNNER)
        self.assertIn("_run_preflight(", RUNNER)
        self.assertIn("preflight_comparison.json", RUNNER)
        self.assertIn('default=lifecycle.parse_gpus("1,2,3")', RUNNER)

    def test_expected_launches_match_fixed_split4_geometry(self):
        self.assertEqual(
            workload.expected_launches(8192, 8176),
            {
                "convert_query": 3,
                "gather": 24,
                "qk": 96,
                "mask": 12,
                "normalize": 24,
                "pv": 96,
                "merge": 24,
            },
        )

    def test_parser_classifies_candidate_only_profile(self):
        value = cell()
        rows = qualification.parse_gpu_rows(profile_log(value))
        result = qualification.qualify(value, rows)
        self.assertTrue(result["qualified"], result)
        self.assertAlmostEqual(result["profile_coverage_ratio"], 1.0)
        self.assertEqual(result["phases"]["qk"]["calls"], 96)
        self.assertEqual(result["phases"]["pv"]["calls"], 96)
        self.assertAlmostEqual(
            result["phases"]["finalize_and_other"]["percent"], 30.0)

    def test_missing_launch_or_numeric_failure_fails_closed(self):
        value = cell()
        value["numerical"]["output_relative_l2"] = 2e-5
        lines = profile_log(value)
        lines[1] = lines[1].replace("  24  ", "  23  ")
        result = qualification.qualify(
            value, qualification.parse_gpu_rows(lines))
        self.assertFalse(result["qualified"])
        self.assertIn(
            "profile workload cell did not qualify", result["reasons"])
        self.assertIn("gather launch count differs", result["reasons"])

    def test_malformed_profiler_output_fails_closed(self):
        result = qualification.qualify(copy.deepcopy(cell()), [])
        self.assertFalse(result["qualified"])
        self.assertIn("ixprof GPU summary is missing", result["reasons"])


if __name__ == "__main__":
    unittest.main()
