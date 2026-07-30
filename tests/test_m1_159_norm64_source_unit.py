import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    ROOT / "qwen3_6_scripts" / "corex_fused_paged_prefill_split4.cu"
).read_text(encoding="utf-8")
SOURCE = (
    ROOT / "qwen3_6_scripts" / "corex_fused_paged_prefill_norm64.cu"
).read_text(encoding="utf-8")
BUILD = (
    ROOT / "qwen3_6_scripts" / "build_corex_fused_paged_prefill_norm64.sh"
).read_text(encoding="utf-8")
PATCH_OPS = (
    ROOT / "qwen3_6_scripts" / "patch_ops.sh"
).read_text(encoding="utf-8")


def _function(source: str, name: str) -> str:
    match = re.search(
        rf"(?:__global__ void|cublasStatus_t) {name}\(.*?\n\}}",
        source,
        re.DOTALL,
    )
    assert match is not None
    return match.group(0)


class M1159Norm64SourceTest(unittest.TestCase):
    def test_one_physical_warp_preserves_256_logical_lanes(self):
        self.assertIn("constexpr int kNormalizeThreads = 64;", SOURCE)
        self.assertIn(
            "constexpr int kNormalizeLogicalLanes = 256;", SOURCE
        )
        self.assertIn(
            "float reduction[kNormalizeLogicalLanes];", SOURCE
        )
        self.assertIn(
            "logical_lane += blockDim.x", SOURCE
        )
        self.assertIn(
            "column += kNormalizeLogicalLanes", SOURCE
        )
        self.assertIn(
            "kNormalizeLogicalLanes / 2", SOURCE
        )
        self.assertIn(
            "normalize_split_scores_kernel<<<rows, kNormalizeThreads",
            SOURCE,
        )

    def test_reduction_operations_and_split_order_are_unchanged(self):
        normalize = _function(SOURCE, "normalize_split_scores_kernel")
        self.assertIn(
            "for (int split = 0; split < active_splits; ++split)",
            normalize,
        )
        self.assertIn("fmaxf(local_max, row_scores[column])", normalize)
        self.assertIn("__fadd_rn(local_sum, probability)", normalize)
        self.assertIn(
            "__fmul_rn(state_sum, correction), reduction[0]", normalize
        )

    def test_qk_pv_and_adjacent_kernels_match_m1_109(self):
        for name in (
            "gather_kv_group_kernel",
            "mask_group_scores_kernel",
            "merge_split_output_kernel",
            "qk_batched",
            "pv_batched",
        ):
            self.assertEqual(_function(SOURCE, name), _function(BASELINE, name))

    def test_build_and_runtime_are_isolated(self):
        self.assertIn(
            "OUTPUT=${VLLM_ROOT}/corex_fused_paged_prefill_norm64.so",
            BUILD,
        )
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill_norm64",
            BUILD,
        )
        self.assertIn(
            '"${SCRIPT_DIR}/corex_fused_paged_prefill_norm64.cu"',
            BUILD,
        )
        self.assertNotIn("corex_fused_paged_prefill_norm64", PATCH_OPS)


if __name__ == "__main__":
    unittest.main()
