from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "qwen3_6_scripts" / "corex_fused_paged_prefill.cu"
BUILD = ROOT / "qwen3_6_scripts" / "build_corex_fused_paged_prefill.sh"
SPLIT_SOURCE = (
    ROOT / "qwen3_6_scripts" / "corex_fused_paged_prefill_split4.cu")
SPLIT_BUILD = (
    ROOT / "qwen3_6_scripts" / "build_corex_fused_paged_prefill_split4.sh")


class FusedPagedPrefillSourceTest(unittest.TestCase):

    def test_fixed_shape_and_streaming_pipeline_are_encoded(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("constexpr int kBlockSize = 16;", source)
        self.assertIn("constexpr int kHeadDim = 256;", source)
        self.assertIn("constexpr int kNumQueryHeads = 6;", source)
        self.assertIn("constexpr int kTileTokens = 512;", source)
        self.assertIn("gather_kv_tile_kernel", source)
        self.assertIn("mask_causal_scores_kernel", source)
        self.assertIn("cublasSgemmStridedBatched", source)
        self.assertIn("at::max(active_scores", source)
        self.assertIn("at::sum(active_scores", source)
        self.assertNotIn("online_softmax_kernel", source)
        self.assertIn("active_blocks.min().item<int>()", source)
        self.assertIn("active_blocks.max().item<int>()", source)
        self.assertIn("out-of-range physical block ID", source)
        self.assertIn("tile_start < context_len", source)
        self.assertIn("key_start < query_len", source)
        self.assertNotIn("torch::matmul", source)

    def test_benchmark_uses_permuted_physical_blocks(self):
        benchmark = (ROOT / "tests" /
                     "bench_fused_paged_prefill_attention.py").read_text(
                         encoding="utf-8")
        self.assertIn("torch.roll(torch.arange", benchmark)
        self.assertIn("key_cache[physical_ids] = logical_key_cache", benchmark)
        self.assertIn(
            "value_cache[physical_ids] = logical_value_cache", benchmark)
        self.assertIn("invalid physical block", benchmark)

    def test_only_qualified_split4_binary_is_hash_pinned(self):
        patch_ops = (ROOT / "qwen3_6_scripts" / "patch_ops.sh").read_text(
            encoding="utf-8")
        manifest = (ROOT / "qwen3_6_scripts" / "prebuilt" /
                    "corex-3.2.3-ivcore10" / "SHA256SUMS").read_text(
                        encoding="utf-8")
        self.assertNotIn("build_corex_fused_paged_prefill.sh", patch_ops)
        self.assertNotIn("build_corex_fused_paged_prefill_split4.sh", patch_ops)
        self.assertIn(
            "e0ff112f965de7126c86a57ba2a64549743ee88c55b25a2396b5f808349ef591  "
            "corex_fused_paged_prefill.so",
            manifest,
        )

    def test_build_uses_fixed_corex_target_and_cublas(self):
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("--cuda-gpu-arch=ivcore10", build)
        self.assertIn("-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill", build)
        self.assertIn("-lcublas", build)

    def test_split4_keeps_fixed_partitions_and_ordered_merge(self):
        source = SPLIT_SOURCE.read_text(encoding="utf-8")
        self.assertIn("constexpr int kTileTokens = 512;", source)
        self.assertIn("constexpr int kSplitCount = 4;", source)
        self.assertIn("constexpr int kGroupTokens = kSplitCount * kTileTokens;",
                      source)
        self.assertIn("gather_kv_group_kernel", source)
        self.assertIn("mask_group_scores_kernel", source)
        self.assertIn("scan_split_max_kernel", source)
        self.assertIn("merge_split_sums_kernel", source)
        self.assertIn("merge_split_output_kernel", source)
        self.assertIn("at::max(active_scores", source)
        self.assertIn("at::sum(active_scores", source)
        self.assertIn("for (int split = 0; split < active_splits; ++split)",
                      source)
        self.assertIn("causal || group_tokens != active_splits * kTileTokens",
                      source)
        self.assertNotIn("kTileTokens = 1024", source)
        self.assertNotIn("kTileTokens = 2048", source)

    def test_split4_build_preserves_native_module_name(self):
        build = SPLIT_BUILD.read_text(encoding="utf-8")
        self.assertIn("--cuda-gpu-arch=ivcore10", build)
        self.assertIn("-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill", build)
        self.assertIn("corex_fused_paged_prefill_split4.cu", build)
        self.assertIn("corex_fused_paged_prefill_split4.so", build)
        self.assertIn("-lcublas", build)

    def test_qualified_split4_has_strict_runtime_dispatch(self):
        paged_attention = (
            ROOT / "qwen3_6_scripts" / "paged_attn.py").read_text(
                encoding="utf-8")
        self.assertIn("_can_use_corex_fused_paged_prefill", paged_attention)
        self.assertIn("query_len <= 16 or query_len > 8192", paged_attention)
        self.assertIn("block_context_len % 16 != 0", paged_attention)
        self.assertIn("block_context_len + query_len > 262144", paged_attention)
        self.assertIn("prefix_key.shape[0] != 0", paged_attention)
        self.assertIn("_PREFIX_BLOCKS_PER_TILE == 32", paged_attention)
        self.assertIn("kv_cache_dtype == \"auto\"", paged_attention)
        self.assertIn("batch_size == 1", paged_attention)
        self.assertIn("_corex_fused_paged_prefill.forward", paged_attention)
        self.assertIn("path={log_key}", paged_attention)

    def test_xformers_passes_explicit_decoder_causality(self):
        patch = (
            ROOT / "qwen3_6_scripts" / "patch_xformers_sdpa_seq.py").read_text(
                encoding="utf-8")
        self.assertIn(
            "is_causal_decoder=(attn_type == AttentionType.DECODER)", patch)
        self.assertIn("_PREFIX_CALL_OLD_BLOCK", patch)
        self.assertIn("_PREFIX_CALL_NEW_BLOCK", patch)


if __name__ == "__main__":
    unittest.main()
