from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from tests import bench_m1_131_exact_sum_softmax as benchmark
from tests import compare_m1_131_exact_sum_softmax as comparator


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_SOURCE = (
    ROOT / "qwen3_6_scripts" / "corex_fused_paged_prefill_exact_sum.cu"
)
FROZEN_M1_109_SOURCE = (
    ROOT / "qwen3_6_scripts" / "corex_fused_paged_prefill_split4.cu"
)
BUILD_SCRIPT = (
    ROOT
    / "qwen3_6_scripts"
    / "build_corex_fused_paged_prefill_exact_sum.sh"
)
RUNNER = ROOT / "scripts" / "run_m1_131_exact_sum_softmax_ab.sh"
FROZEN_M1_109_SHA256 = (
    "11c387e6012834fe634ffa8d038f7a4bf"
    "4ec19fa13ec23779ee1f414037e564b"
)
CONTROL_SHA = benchmark.CONTROL_EXTENSION_SHA256
CANDIDATE_SHA = "2" * 64


def valid_cell(
    case_name: str,
    *,
    speedup: float = 1.25,
    gpu: int = 0,
) -> dict:
    context_len, query_len, kind = benchmark.CASES[case_name]
    control_ms = 10.0
    candidate_ms = control_ms / speedup
    return {
        "schema": benchmark.SCHEMA,
        "source_commit": "a" * 40,
        "runtime_identity": "corex-3.2.3-m1-131-exact-sum",
        "instance": "unit",
        "visible_physical_gpu": gpu,
        "device_name": "BI-V100",
        "torch_version": "unit",
        "case": case_name,
        "kind": kind,
        "context_len": context_len,
        "query_len": query_len,
        "total_kv_len": context_len + query_len,
        "seed": benchmark.production.SEED,
        "warmups": benchmark.WARMUPS,
        "trials": benchmark.TRIALS,
        "physical_block_permutation": context_len > 0,
        "extensions": {
            "control": {
                "path": "/tmp/control.so",
                "size_bytes": 1,
                "sha256": CONTROL_SHA,
            },
            "candidate": {
                "path": "/tmp/candidate.so",
                "size_bytes": 1,
                "sha256": CANDIDATE_SHA,
            },
        },
        "output_contract": {
            "control_result_arity_ok": True,
            "candidate_result_arity_ok": True,
            "candidate_repeat_contract_ok": True,
            "control_output_shape_ok": True,
            "candidate_output_shape_ok": True,
            "control_lse_shape_ok": True,
            "candidate_lse_shape_ok": True,
            "control_output_dtype_ok": True,
            "candidate_output_dtype_ok": True,
            "control_lse_dtype_ok": True,
            "candidate_lse_dtype_ok": True,
            "control_device_ok": True,
            "candidate_device_ok": True,
            "control_contiguous": True,
            "candidate_contiguous": True,
        },
        "timings": {
            "control": {
                "warmups": benchmark.WARMUPS,
                "trials": benchmark.TRIALS,
                "cuda_trials_ms": [control_ms] * benchmark.TRIALS,
                "cuda_median_ms": control_ms,
                "host_trials_ms": [control_ms] * benchmark.TRIALS,
                "host_median_ms": control_ms,
            },
            "candidate": {
                "warmups": benchmark.WARMUPS,
                "trials": benchmark.TRIALS,
                "cuda_trials_ms": [candidate_ms] * benchmark.TRIALS,
                "cuda_median_ms": candidate_ms,
                "host_trials_ms": [candidate_ms] * benchmark.TRIALS,
                "host_median_ms": candidate_ms,
            },
            "control_over_candidate_speedup": speedup,
        },
        "numerical": {
            "control_finite": True,
            "candidate_finite": True,
            "output_exact": True,
            "lse_exact": True,
            "candidate_repeat_output_exact": True,
            "candidate_repeat_lse_exact": True,
            "output_relative_l2": 0.0,
            "lse_relative_l2": 0.0,
            "output_max_abs": 0.0,
            "lse_max_abs": 0.0,
        },
        "authorization": {
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }


class M1131ExactSumSoftmaxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.candidate_source = CANDIDATE_SOURCE.read_text(encoding="utf-8")
        cls.build_source = BUILD_SCRIPT.read_text(encoding="utf-8")
        cls.runner_source = RUNNER.read_text(encoding="utf-8")

    def test_m1_109_authoritative_source_is_frozen(self):
        digest = hashlib.sha256(FROZEN_M1_109_SOURCE.read_bytes()).hexdigest()
        self.assertEqual(digest, FROZEN_M1_109_SHA256)

    def test_candidate_preserves_authoritative_sum_order(self):
        source = self.candidate_source
        self.assertIn("normalize_scores_exact_sum_kernel", source)
        self.assertIn("auto split_sums = at::sum(active_scores", source)
        self.assertIn("merge_split_sums_kernel<<<", source)
        self.assertIn("__fadd_rn(", source)
        self.assertIn("__fmul_rn(", source)
        self.assertNotIn("at::max(active_scores", source)
        self.assertNotIn("scan_split_max_kernel", source)

    def test_fused_kernel_does_not_update_running_sum(self):
        source = self.candidate_source
        start = source.index("__global__ void normalize_scores_exact_sum_kernel")
        end = source.index("__global__ void merge_split_sums_kernel", start)
        kernel = source[start:end]
        self.assertNotIn("running_sum", kernel)
        self.assertIn("__fsub_rn(state_max, next_max)", kernel)
        self.assertIn("__fsub_rn(score, next_max)", kernel)
        self.assertIn("running_max[row] = state_max", kernel)

    def test_candidate_scope_and_fallback_boundaries_remain_fixed(self):
        source = self.candidate_source
        for fragment in (
            "constexpr int kBlockSize = 16;",
            "constexpr int kHeadDim = 256;",
            "constexpr int kNumQueryHeads = 4;",
            "constexpr int kNumKvHeads = 1;",
            "constexpr int kMaxQueryTokens = 8192;",
            "constexpr int kMaxSequenceTokens = 262144;",
            "query must have shape (Q, 4, 256)",
            "key_new must have shape (Q, 1, 256)",
            "auto float_options = query.options().dtype(torch::kFloat32);",
        ):
            self.assertIn(fragment, source)

    def test_build_uses_a_unique_module_and_ivcore10(self):
        source = self.build_source
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill_exact_sum",
            source,
        )
        self.assertIn("--cuda-gpu-arch=ivcore10", source)
        self.assertIn("corex_fused_paged_prefill_exact_sum.so", source)
        self.assertNotIn("corex_fused_paged_prefill_split4.so", source)

    def test_runner_has_fixed_matrix_and_scoped_lifecycle(self):
        source = self.runner_source
        for case_name in benchmark.CASES:
            self.assertIn(case_name, source)
        for fragment in (
            "bi100_stop_process_group",
            " 60 20",
            "service_postflight_gate.py",
            "bi100_preflight.py",
            "compare_bi100_preflights.py",
            "fatal_scan.rc",
            "timeout --foreground --signal=TERM --kill-after=60s 3600s",
            "status --porcelain --untracked-files=all",
            "reason=read_error",
            "fatal_scan_find.rc",
            "post_wait_cleanup_rc",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("pkill", source)
        self.assertNotIn("kill -9", source)

    def test_valid_exact_cell_passes(self):
        cell = valid_cell("production_65k_q8176")
        self.assertEqual(
            benchmark.evaluate_cell(cell),
            {"qualified": True, "reasons": []},
        )

    def test_output_or_repeat_drift_fails_closed(self):
        cell = valid_cell("production_128k_q8176")
        cell["numerical"]["output_exact"] = False
        cell["numerical"]["candidate_repeat_lse_exact"] = False
        result = benchmark.evaluate_cell(cell)
        self.assertFalse(result["qualified"])
        self.assertTrue(
            any("output_exact" in reason for reason in result["reasons"])
        )
        self.assertTrue(
            any(
                "candidate_repeat_lse_exact" in reason
                for reason in result["reasons"]
            )
        )

    def test_wrong_output_contract_or_control_binary_fails_closed(self):
        cell = valid_cell("production_65k_q8176")
        cell["output_contract"]["candidate_output_shape_ok"] = False
        cell["extensions"]["control"]["sha256"] = "3" * 64
        result = benchmark.evaluate_cell(cell)
        self.assertFalse(result["qualified"])
        self.assertTrue(
            any("candidate_output_shape_ok" in reason
                for reason in result["reasons"])
        )
        self.assertTrue(
            any("frozen M1-108" in reason for reason in result["reasons"])
        )

    def test_four_exact_faster_cells_authorize_only_tp4_experiment(self):
        cells = [
            valid_cell(name, gpu=gpu)
            for gpu, name in enumerate(benchmark.CASES)
        ]
        report = comparator.compare(cells)
        self.assertTrue(report["qualified"], report["reasons"])
        self.assertEqual(report["positive_cases"], 4)
        self.assertEqual(
            report["decision"],
            {
                "tp4_service_experiment_authorized": True,
                "main_or_yaml_change_authorized": False,
                "official_score_claim_authorized": False,
            },
        )

    def test_aggregate_rejects_regression_and_malformed_cell(self):
        cells = [
            valid_cell(name, gpu=gpu)
            for gpu, name in enumerate(benchmark.CASES)
        ]
        cells[0] = valid_cell(
            next(iter(benchmark.CASES)), speedup=0.95, gpu=0
        )
        report = comparator.compare(cells)
        self.assertFalse(report["qualified"])
        self.assertTrue(
            any("regressed" in reason for reason in report["reasons"])
        )

        malformed = deepcopy(cells)
        malformed[0] = {"schema": benchmark.SCHEMA}
        report = comparator.compare(malformed)
        self.assertFalse(report["qualified"])
        self.assertTrue(report["reasons"])

    def test_json_loader_fails_closed_without_exception_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.json"
            bad = root / "bad.json"
            good.write_text(
                json.dumps(valid_cell("production_dense_q8176")),
                encoding="utf-8",
            )
            bad.write_text("{private malformed value", encoding="utf-8")
            cells, errors = comparator.load_cells([good, bad])
        self.assertIsInstance(cells[0], dict)
        self.assertIsNone(cells[1])
        self.assertEqual(
            errors, ["cell[1] could not be loaded as JSON"]
        )
        self.assertNotIn("private", errors[0])


if __name__ == "__main__":
    unittest.main()
