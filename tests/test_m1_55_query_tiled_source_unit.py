import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "qwen3_6_scripts" / "corex_query_tiled_paged_prefill.cu"
).read_text(encoding="utf-8")
BUILD = (
    ROOT / "tests" / "build_corex_query_tiled_paged_prefill.sh"
).read_text(encoding="utf-8")
PATCH_OPS = (
    ROOT / "qwen3_6_scripts" / "patch_ops.sh"
).read_text(encoding="utf-8")


class M155QueryTiledSourceUnitTest(unittest.TestCase):
    def test_fixed_production_abi(self):
        for text in (
            "constexpr int kBlockSize = 16;",
            "constexpr int kHeadDim = 256;",
            "constexpr int kNumQueryHeads = 4;",
            "constexpr int kNumKvHeads = 1;",
            "constexpr int kQueryTile = 16;",
            "constexpr int kKeyTile = 16;",
            "constexpr int kReductionTokens = 512;",
            "constexpr int kWarpSize = 64;",
            "constexpr int kMaxSequenceTokens = 262144;",
        ):
            self.assertIn(text, SOURCE)

    def test_direct_page_table_reads_are_present(self):
        self.assertIn("block_table[logical_block]", SOURCE)
        self.assertIn("key_cache[index]", SOURCE)
        self.assertIn("value_cache[index]", SOURCE)
        self.assertIn("logical_token < context_len", SOURCE)

    def test_wmma_covers_qk_and_pv(self):
        self.assertIn("#include <mma.h>", SOURCE)
        self.assertGreaterEqual(SOURCE.count("wmma::mma_sync("), 2)
        self.assertIn("query_fragments[kDimTiles]", SOURCE)
        self.assertIn("probability_fragment", SOURCE)
        self.assertIn("value_fragment", SOURCE)

    def test_online_softmax_state_is_fp32_and_local(self):
        self.assertIn("float running_output[kQueryTile * kHeadDim]", SOURCE)
        self.assertIn("float running_max[kQueryTile]", SOURCE)
        self.assertIn("float running_sum[kQueryTile]", SOURCE)
        self.assertIn("expf(score - new_max)", SOURCE)
        self.assertIn("logf(shared.running_sum", SOURCE)
        self.assertIn(
            "phase == 0 ? context_len : last_query", SOURCE)

    def test_no_full_query_global_intermediates(self):
        self.assertNotIn("split_output", SOURCE)
        self.assertNotIn("converted_query", SOURCE)
        self.assertNotIn("cublas", SOURCE.lower())
        self.assertNotIn("at::max", SOURCE)
        self.assertNotIn("at::sum", SOURCE)
        self.assertEqual(SOURCE.count("torch::empty"), 2)

    def test_causal_and_capacity_guards_are_explicit(self):
        self.assertIn("logical_key <= absolute_query", SOURCE)
        self.assertIn("context_len must be block aligned", SOURCE)
        self.assertIn("context_len + query_len exceeds 262144", SOURCE)
        self.assertIn("block_table contains an out-of-range", SOURCE)

    def test_candidate_is_not_installed_or_enabled(self):
        self.assertNotIn("corex_query_tiled_paged_prefill", PATCH_OPS)
        self.assertNotIn("build_corex_query_tiled_paged_prefill", PATCH_OPS)

    def test_build_is_isolated_and_ivcore10_bound(self):
        self.assertIn("--cuda-gpu-arch=ivcore10", BUILD)
        self.assertIn("-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill", BUILD)
        self.assertIn("corex_query_tiled_paged_prefill.cu", BUILD)


if __name__ == "__main__":
    unittest.main()
