from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "qwen3_6_scripts" / "corex_fused_paged_prefill.cu"
BUILD = ROOT / "qwen3_6_scripts" / "build_corex_fused_paged_prefill.sh"


class FusedPagedPrefillSourceTest(unittest.TestCase):

    def test_fixed_shape_and_streaming_pipeline_are_encoded(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("constexpr int kBlockSize = 16;", source)
        self.assertIn("constexpr int kHeadDim = 256;", source)
        self.assertIn("constexpr int kNumQueryHeads = 6;", source)
        self.assertIn("constexpr int kTileTokens = 512;", source)
        self.assertIn("gather_kv_tile_kernel", source)
        self.assertIn("online_softmax_kernel", source)
        self.assertIn("cublasSgemmStridedBatched", source)
        self.assertIn("tile_start < context_len", source)
        self.assertIn("key_start < query_len", source)
        self.assertIn("column0 >= valid_tokens", source)
        self.assertNotIn("torch::matmul", source)

    def test_candidate_is_not_installed_before_qualification(self):
        patch_ops = (ROOT / "qwen3_6_scripts" / "patch_ops.sh").read_text(
            encoding="utf-8")
        manifest = (ROOT / "qwen3_6_scripts" / "prebuilt" /
                    "corex-3.2.3-ivcore10" / "SHA256SUMS").read_text(
                        encoding="utf-8")
        self.assertNotIn("build_corex_fused_paged_prefill.sh", patch_ops)
        self.assertNotIn("corex_fused_paged_prefill.so", manifest)

    def test_build_uses_fixed_corex_target_and_cublas(self):
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("--cuda-gpu-arch=ivcore10", build)
        self.assertIn("-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill", build)
        self.assertIn("-lcublas", build)


if __name__ == "__main__":
    unittest.main()
