from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "tests/corex_fused_paged_prefill_grouped_qk_split512_ext.cu")
BUILD = ROOT / "tests/build_corex_fused_prefill_grouped_qk_split512.sh"
COMPONENT_RUNNER = ROOT / "scripts/run_m1_109_fused_softmax_component_ab.sh"
WRAPPER = (
    ROOT / "scripts/run_m1_126_grouped_qk_split512_component_ab.sh")


class GroupedQkSplit512SourceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_fixed_production_shape_and_precision_contract(self) -> None:
        for marker in (
            "constexpr int kBlockSize = 16;",
            "constexpr int kHeadDim = 256;",
            "constexpr int kNumQueryHeads = 4;",
            "constexpr int kNumKvHeads = 1;",
            "constexpr int kTileTokens = 512;",
            "constexpr int kSplitCount = 4;",
            "constexpr int kGroupTokens = kSplitCount * kTileTokens;",
            "query.options().dtype(torch::kFloat32)",
            "context_len + query_len exceeds 262144",
        ):
            self.assertIn(marker, self.source)
        self.assertNotIn("cublasGemmEx", self.source)
        self.assertNotIn("CUBLAS_COMPUTE_16F", self.source)

    def test_one_grouped_qk_retains_split512_softmax_and_pv(self) -> None:
        self.assertEqual(self.source.count("check_cublas(qk_group("), 1)
        self.assertIn(
            "padded_tokens, rows, kHeadDim,\n"
            "      &alpha, key_tile, kHeadDim,\n"
            "      query, kHeadDim,\n"
            "      &beta, scores, kGroupTokens",
            self.source,
        )
        self.assertIn(
            "for (int split = 0; split < active_splits; ++split)",
            self.source,
        )
        self.assertIn("normalize_split_scores_kernel", self.source)
        self.assertIn("split * kTileTokens", self.source)
        self.assertIn(
            "kHeadDim, rows, kTileTokens,\n"
            "      &alpha, value_tile, kHeadDim,\n"
            "      scores, kGroupTokens",
            self.source,
        )
        self.assertIn("merge_split_output_kernel", self.source)
        self.assertNotIn("cublasSgemmStridedBatched", self.source)

    def test_tail_and_causal_columns_are_masked(self) -> None:
        for marker in (
            "token_offset >= group_tokens",
            "column >= group_tokens",
            "causal && logical_token > context_len + query_index",
            "const int padded_tokens = active_splits * kTileTokens;",
            "causal || group_tokens != padded_tokens",
        ):
            self.assertIn(marker, self.source)

    def test_bounded_workspace_and_safe_failure_are_retained(self) -> None:
        self.assertIn(
            "{kNumQueryHeads, query_len, kGroupTokens}", self.source)
        self.assertIn(
            "{kSplitCount, kNumQueryHeads, query_len, kHeadDim}",
            self.source,
        )
        self.assertIn(
            "block_table contains an out-of-range physical block ID",
            self.source,
        )
        self.assertIn("C10_CUDA_KERNEL_LAUNCH_CHECK()", self.source)
        self.assertNotIn(
            "kMaxSequenceTokens, kMaxSequenceTokens", self.source)

    def test_build_and_runner_scripts_are_valid(self) -> None:
        build = BUILD.read_text(encoding="utf-8")
        wrapper = WRAPPER.read_text(encoding="utf-8")
        runner = COMPONENT_RUNNER.read_text(encoding="utf-8")
        self.assertIn("--cuda-gpu-arch=ivcore10", build)
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill", build)
        self.assertIn(
            "corex_fused_paged_prefill_grouped_qk_split512_ext.cu", build)
        self.assertIn(
            "BI100_COMPONENT_AB_VARIANT=m1-126-grouped-qk-split512",
            wrapper,
        )
        self.assertIn(
            '"bi100-m1-126-grouped-qk-split512-component-ab-v1"',
            runner,
        )
        self.assertIn("median_speedup < 1.10", runner)
        self.assertIn("min(speedups) < 1.0 / 1.02", runner)
        for path in (BUILD, WRAPPER, COMPONENT_RUNNER):
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runner_embedded_python_remains_valid(self) -> None:
        runner = COMPONENT_RUNNER.read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", runner, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 3)
        for index, block in enumerate(blocks):
            compile(block, f"<runner-heredoc-{index}>", "exec")


if __name__ == "__main__":
    unittest.main()
