from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = ROOT / "qwen3_6_scripts/corex_fused_paged_prefill_split4.cu"
CANDIDATE_SOURCE = (
    ROOT / "qwen3_6_scripts/corex_fused_paged_prefill_half_qk.cu"
)
DEFAULT_BUILD = (
    ROOT / "qwen3_6_scripts/build_corex_fused_paged_prefill_split4.sh"
)
CANDIDATE_BUILD = (
    ROOT / "qwen3_6_scripts/build_corex_fused_paged_prefill_half_qk.sh"
)


class M1128HalfQkPipelineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_source = BASE_SOURCE.read_text(encoding="utf-8")
        cls.candidate_source = CANDIDATE_SOURCE.read_text(encoding="utf-8")
        cls.default_build = DEFAULT_BUILD.read_text(encoding="utf-8")
        cls.candidate_build = CANDIDATE_BUILD.read_text(encoding="utf-8")

    def test_frozen_default_source_is_unchanged(self) -> None:
        digest = hashlib.sha256(BASE_SOURCE.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "11c387e6012834fe634ffa8d038f7a4bf4ec19fa13ec23779ee1f414037e564b",
        )
        self.assertNotIn("BI100_HALF_INPUT_QK", self.base_source)
        self.assertIn("cublasSgemmStridedBatched(", self.base_source)
        self.assertNotIn("-DBI100_HALF_INPUT_QK=1", self.default_build)
        self.assertIn(
            "corex_fused_paged_prefill_split4.cu", self.default_build)
        self.assertNotIn(
            "corex_fused_paged_prefill_half_qk.cu", self.default_build)

    def test_candidate_changes_only_qk_input_pipeline(self) -> None:
        for marker in (
            "#define BI100_HALF_INPUT_QK 1",
            "using KeyTile = __half;",
            "prepared[index] = query[source];",
            "key_value = key_cache[key_index];",
            "key_value = key_new[source];",
            "cublasGemmStridedBatchedEx(",
            "key_tile, CUDA_R_16F",
            "query, CUDA_R_16F",
            "scores, CUDA_R_32F",
            "kNumQueryHeads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP",
        ):
            self.assertIn(marker, self.candidate_source)
        self.assertIn("float* value_tiles", self.candidate_source)
        self.assertIn("cublasSgemmStridedBatched(", self.candidate_source)

    def test_candidate_build_is_separate_and_fixed(self) -> None:
        for marker in (
            "-DBI100_HALF_INPUT_QK=1",
            "-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill",
            "corex_fused_paged_prefill_half_qk.cu",
            "corex_fused_paged_prefill_half_qk.so",
            "--cuda-gpu-arch=ivcore10",
        ):
            self.assertIn(marker, self.candidate_build)
        completed = subprocess.run(
            ["bash", "-n", str(CANDIDATE_BUILD)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_softmax_pv_merge_and_output_contract_remain_present(self) -> None:
        for marker in (
            "normalize_split_scores_kernel<<<rows",
            "check_cublas(pv_batched(",
            "merge_split_output_kernel<<<",
            "running_output.div_(running_sum.unsqueeze(-1));",
            ".to(query.scalar_type()).contiguous();",
            "return {output, lse};",
        ):
            self.assertIn(marker, self.candidate_source)


if __name__ == "__main__":
    unittest.main()
