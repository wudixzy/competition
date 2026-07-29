from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/corex_fused_paged_prefill_group2048_ext.cu"
BUILD = ROOT / "tests/build_corex_fused_prefill_group2048.sh"
COMPONENT_RUNNER = ROOT / "scripts/run_m1_109_fused_softmax_component_ab.sh"
WRAPPER = ROOT / "scripts/run_m1_113_group2048_component_ab.sh"


class Group2048PrefillSourceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_fixed_production_shape_and_fp32_contract(self):
        self.assertIn("constexpr int kBlockSize = 16;", self.source)
        self.assertIn("constexpr int kHeadDim = 256;", self.source)
        self.assertIn("constexpr int kNumQueryHeads = 4;", self.source)
        self.assertIn("constexpr int kNumKvHeads = 1;", self.source)
        self.assertIn("constexpr int kGroupTokens = 2048;", self.source)
        self.assertIn("query.options().dtype(torch::kFloat32)", self.source)
        self.assertIn("cublasSgemm(", self.source)
        self.assertNotIn("cublasGemmEx", self.source)
        self.assertNotIn("CUBLAS_COMPUTE_16F", self.source)

    def test_gqa_heads_are_flattened_into_one_qk_and_one_pv(self):
        self.assertIn(
            "group_tokens, rows, kHeadDim,\n"
            "      &alpha, key_tile, kHeadDim,\n"
            "      query, kHeadDim,\n"
            "      &beta, scores, kGroupTokens",
            self.source,
        )
        self.assertIn(
            "kHeadDim, rows, group_tokens,\n"
            "      &alpha, value_tile, kHeadDim,\n"
            "      scores, kGroupTokens,\n"
            "      &beta, output, kHeadDim",
            self.source,
        )
        self.assertEqual(self.source.count("check_cublas(qk_group("), 1)
        self.assertEqual(self.source.count("check_cublas(pv_group("), 1)
        self.assertNotIn("cublasSgemmStridedBatched", self.source)

    def test_scores_keep_a_stable_2048_leading_dimension(self):
        self.assertIn(
            "{kNumQueryHeads, query_len, kGroupTokens}", self.source)
        self.assertIn(
            "scores + static_cast<int64_t>(row) * kGroupTokens",
            self.source,
        )
        self.assertIn(
            "scores[static_cast<int64_t>(row) * kGroupTokens + column]",
            self.source,
        )

    def test_online_softmax_does_not_materialize_full_sequence_logits(self):
        self.assertIn("normalize_group_scores_kernel", self.source)
        self.assertIn("running_max", self.source)
        self.assertIn("running_sum", self.source)
        self.assertIn("accumulate_output_kernel", self.source)
        self.assertNotIn("kMaxSequenceTokens, kMaxSequenceTokens",
                         self.source)

    def test_shape_checks_and_safe_failure_are_retained(self):
        self.assertIn(
            "query must have shape (Q, 4, 256)", self.source)
        self.assertIn(
            "context_len + query_len exceeds 262144", self.source)
        self.assertIn(
            "block_table contains an out-of-range physical block ID",
            self.source,
        )
        self.assertIn("C10_CUDA_KERNEL_LAUNCH_CHECK()", self.source)

    def test_build_script_is_syntactically_valid_and_targets_ivcore10(self):
        source = BUILD.read_text(encoding="utf-8")
        self.assertEqual(
            hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "74a0ca7551cd567217971ec6bbf2ed1507ca829f1bfecbfa2c60bca484f5fcae",
        )
        self.assertEqual(
            hashlib.sha256(BUILD.read_bytes()).hexdigest(),
            "0ca153b0415a6085bc0336c0715ea90cbfce6692e7c1c401e0cda16ee4d7c161",
        )
        self.assertIn("--cuda-gpu-arch=ivcore10", source)
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill", source)
        self.assertIn(
            "corex_fused_paged_prefill_group2048_ext.cu", source)
        completed = subprocess.run(
            ["bash", "-n", str(BUILD)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_component_ab_reuses_the_fixed_four_case_lifecycle(self):
        source = COMPONENT_RUNNER.read_text(encoding="utf-8")
        wrapper = WRAPPER.read_text(encoding="utf-8")
        self.assertIn(
            "BI100_COMPONENT_AB_VARIANT=m1-113-group2048", wrapper)
        self.assertIn(
            'exec "$ROOT/scripts/run_m1_109_fused_softmax_component_ab.sh" '
            '"$@"',
            wrapper,
        )
        self.assertIn(
            '"bi100-m1-113-group2048-component-ab-v1"', source)
        for case in (
            "production_dense_q8176",
            "production_65k_q8176",
            "production_128k_q8176",
            "production_235k_q5616",
        ):
            self.assertIn(case, source)
        self.assertIn("median_speedup < 1.10", source)
        self.assertIn("min(speedups) < 1.0 / 1.02", source)
        self.assertIn("service_postflight_gate.py", source)
        self.assertIn("bi100_preflight.py", source)
        self.assertIn("exec_bi100_session.py", source)
        self.assertIn("cleanup_recorded_bi100_sessions.py", source)
        self.assertIn("qualify_recorded_session_cleanup.py", source)
        self.assertIn("session_recovery_clean.rc", source)
        self.assertIn("timeout_scan.rc", source)
        self.assertIn('"bi100-component-ab-runner-status-v1"', source)
        self.assertIn('wait "${PIDS[$gpu]}"', source)
        self.assertNotIn("setsid ", source)
        self.assertNotIn("pkill", source)
        self.assertNotIn("killall", source)
        for path in (COMPONENT_RUNNER, WRAPPER):
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
