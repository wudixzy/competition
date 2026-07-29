from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
M1_128_SOURCE = (
    ROOT / "qwen3_6_scripts/corex_fused_paged_prefill_half_qk.cu"
)
M1_129_SOURCE = (
    ROOT / "qwen3_6_scripts/corex_fused_paged_prefill_half_qk_default.cu"
)
BUILD = (
    ROOT
    / "qwen3_6_scripts/build_corex_fused_paged_prefill_half_qk_default.sh"
)


class M1129HalfQkDefaultGemmExTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.m1_128 = M1_128_SOURCE.read_text(encoding="utf-8")
        cls.m1_129 = M1_129_SOURCE.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")

    def test_only_gemmex_algorithm_differs_from_m1_128(self) -> None:
        self.assertIn(
            "CUBLAS_GEMM_DEFAULT_TENSOR_OP", self.m1_128)
        self.assertNotIn(
            "CUBLAS_GEMM_DEFAULT_TENSOR_OP", self.m1_129)
        self.assertIn(
            "kNumQueryHeads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);",
            self.m1_129,
        )
        expected = self.m1_128.replace(
            "CUBLAS_GEMM_DEFAULT_TENSOR_OP",
            "CUBLAS_GEMM_DEFAULT",
        )
        self.assertEqual(self.m1_129, expected)

    def test_candidate_retains_half_input_fp32_accumulation(self) -> None:
        for marker in (
            "#define BI100_HALF_INPUT_QK 1",
            "using KeyTile = __half;",
            "key_tile, CUDA_R_16F",
            "query, CUDA_R_16F",
            "scores, CUDA_R_32F",
            "kNumQueryHeads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT",
            "float* value_tiles",
            "running_output.div_(running_sum.unsqueeze(-1));",
            "return {output, lse};",
        ):
            self.assertIn(marker, self.m1_129)

    def test_build_is_separate_and_syntactically_valid(self) -> None:
        for marker in (
            "corex_fused_paged_prefill_half_qk_default.cu",
            "corex_fused_paged_prefill_half_qk_default.so",
            "-DBI100_HALF_INPUT_QK=1",
            "--cuda-gpu-arch=ivcore10",
        ):
            self.assertIn(marker, self.build)
        completed = subprocess.run(
            ["bash", "-n", str(BUILD)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
