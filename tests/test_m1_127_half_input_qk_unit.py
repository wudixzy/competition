from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/corex_half_input_qk_gemm_ext.cu"
BUILD = ROOT / "tests/build_corex_half_input_qk_gemm.sh"
BENCH = ROOT / "tests/bench_m1_127_half_input_qk.py"
RUNNER = ROOT / "scripts/run_m1_127_half_input_qk.sh"


class M1127HalfInputQkUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.bench = BENCH.read_text(encoding="utf-8")

    def test_candidate_keeps_model_dtype_and_fp32_accumulation(self) -> None:
        for marker in (
            "query.scalar_type() == torch::kFloat16",
            "key.scalar_type() == torch::kFloat16",
            "CUDA_R_16F, kHeadDim",
            "output.data_ptr<float>(), CUDA_R_32F",
            "kHeads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP",
        ):
            self.assertIn(marker, self.source)
        self.assertNotIn("CUDA_R_16BF", self.source)
        self.assertNotIn("CUBLAS_COMPUTE_16F", self.source)

    def test_control_matches_m1_109_qk_submission_shape(self) -> None:
        for marker in (
            "cublasSgemmStridedBatched",
            "kTileTokens, query_tokens, kHeadDim",
            "key.data_ptr<float>(), kHeadDim, 0",
            "static_cast<long long>(query_tokens) * kHeadDim",
            "static_cast<long long>(query_tokens) * kTileTokens",
            "kHeads",
        ):
            self.assertIn(marker, self.source)

    def test_benchmark_uses_fixed_shapes_and_independent_oracle(self) -> None:
        for marker in (
            '"q8176": 8176',
            '"q5616": 5616',
            "MAGNITUDES = (0.5, 1.0, 2.0)",
            "SAMPLED_QUERIES = 16",
            "candidate_vs_fp64_sample",
            "actual.detach().cpu().double()",
            "expected.detach().cpu().double()",
            "timed_candidate_vs_control_max_abs",
            "RELATIVE_L2_LIMIT = 1e-5",
            "MAX_ABS_LIMIT = 1e-3",
            "MIN_SPEEDUP = 1.25",
            '"full_pipeline_integration_authorized": not reasons',
            '"tp4_service_authorized": False',
        ):
            self.assertIn(marker, self.bench)
        ast.parse(self.bench)

    def test_build_script_targets_corex_ivcore10(self) -> None:
        source = BUILD.read_text(encoding="utf-8")
        self.assertIn("--cuda-gpu-arch=ivcore10", source)
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_half_input_qk_gemm", source)
        self.assertIn("corex_half_input_qk_gemm_ext.cu", source)
        completed = subprocess.run(
            ["bash", "-n", str(BUILD)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runner_freezes_two_cells_and_scoped_lifecycle(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for marker in (
            "CASES=(q8176 q5616)",
            "GPUS=(0 1)",
            "exec_bi100_session.py",
            "bi100_stop_process_group",
            '"$pgid" "$leader" 60 20',
            "bench_m1_127_half_input_qk.py",
            "--gpus 0,1,2,3",
            "service_postflight_gate.py",
            "compare_bi100_preflights.py",
            "fatal_scan.rc",
            "timeout_scan.rc",
            '"full_pipeline_integration_authorized": qualified',
            '"tp4_service_authorized": False',
        ):
            self.assertIn(marker, source)
        self.assertNotIn("pkill", source)
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
