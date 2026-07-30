import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "qwen3_6_scripts" / "corex_fused_paged_prefill_fp16_qk.cu"
).read_text(encoding="utf-8")
BUILD = (
    ROOT / "qwen3_6_scripts" / "build_corex_fused_paged_prefill_fp16_qk.sh"
).read_text(encoding="utf-8")


class M1157Fp16QkSourceTest(unittest.TestCase):
    def test_qk_uses_fp16_inputs_with_fp32_accumulation(self):
        match = re.search(
            r"cublasStatus_t qk_batched\(.*?\n\}", SOURCE, re.DOTALL
        )
        self.assertIsNotNone(match)
        qk = match.group(0)
        self.assertIn("cublasGemmStridedBatchedEx", qk)
        self.assertEqual(qk.count("CUDA_R_16F"), 2)
        self.assertEqual(qk.count("CUDA_R_32F"), 2)
        self.assertIn("CUBLAS_GEMM_DEFAULT_TENSOR_OP", qk)
        self.assertNotIn("cublasSgemmStridedBatched", qk)

    def test_candidate_keeps_pv_and_online_softmax_in_fp32(self):
        self.assertIn(
            "normalize_split_scores_kernel", SOURCE
        )
        self.assertIn(
            "cublasStatus_t pv_batched(\n"
            "    cublasHandle_t handle, const float* value_tile, "
            "const float* scores",
            SOURCE,
        )
        self.assertIn("return cublasSgemmStridedBatched(", SOURCE)
        self.assertIn(
            "auto running_output = torch::zeros(\n"
            "      {kNumQueryHeads, query_len, kHeadDim}, float_options);",
            SOURCE,
        )

    def test_only_q_and_k_staging_are_fp16(self):
        self.assertIn(
            "auto packed_query = torch::empty(\n"
            "      {kNumQueryHeads, query_len, kHeadDim}, half_options);",
            SOURCE,
        )
        self.assertIn(
            "auto key_tiles = torch::empty(\n"
            "      {kSplitCount, kTileTokens, kHeadDim}, half_options);",
            SOURCE,
        )
        self.assertIn(
            "auto value_tiles = torch::empty(\n"
            "      {kSplitCount, kTileTokens, kHeadDim}, float_options);",
            SOURCE,
        )

    def test_build_isolated_from_production_extension(self):
        self.assertIn(
            "OUTPUT=${VLLM_ROOT}/corex_fused_paged_prefill_fp16_qk.so",
            BUILD,
        )
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill_fp16_qk",
            BUILD,
        )
        self.assertIn(
            '"${SCRIPT_DIR}/corex_fused_paged_prefill_fp16_qk.cu"',
            BUILD,
        )
        self.assertNotIn("corex_fused_paged_prefill_split4.cu", BUILD)


if __name__ == "__main__":
    unittest.main()
