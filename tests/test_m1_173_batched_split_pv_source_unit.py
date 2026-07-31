from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "qwen3_6_scripts" / "corex_fused_paged_prefill_fp16_qk.cu")
BUILD = (
    ROOT / "qwen3_6_scripts"
    / "build_corex_fused_paged_prefill_batched_split_pv.sh")
RUNNER = ROOT / "scripts" / "run_m1_173_batched_split_pv_ab.py"
PATCH_OPS = ROOT / "qwen3_6_scripts" / "patch_ops.sh"
DEFAULT_BUILD = (
    ROOT / "qwen3_6_scripts" / "build_corex_fused_paged_prefill_fp16_qk.sh")


class M1173BatchedSplitPvSourceTest(unittest.TestCase):
    def test_candidate_batches_split_and_head_without_precision_change(self):
        source = SOURCE.read_text(encoding="utf-8")
        for fragment in (
            "#if defined(BI100_BATCHED_SPLIT_PV)",
            "pv_split_head_batched(",
            "cublasSgemmStridedBatched(",
            "active_splits * kNumQueryHeads",
            "{kSplitCount, kNumQueryHeads, kTileTokens, kHeadDim}",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)
        self.assertIn("const float* value_tiles", source)
        self.assertNotIn("CUDA_R_16F, kHeadDim,", source.split(
            "pv_split_head_batched(", 1)[1].split("#endif", 1)[0])

    def test_default_path_and_production_overlay_remain_unchanged(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("#else\n    value_tiles[index] = value_value;", source)
        self.assertIn("#else\n    for (int split = 0;", source)
        self.assertNotIn(
            "corex_fused_paged_prefill_batched_split_pv",
            PATCH_OPS.read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            "BI100_BATCHED_SPLIT_PV",
            DEFAULT_BUILD.read_text(encoding="utf-8"),
        )

    def test_build_and_runner_are_explicit_and_fail_closed(self):
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("-DBI100_BATCHED_SPLIT_PV=1", build)
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill_batched_split_pv",
            build,
        )
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("bench_m1_162_calibrated_fp16_qk_ab.py", runner)
        self.assertIn("bi100-m1-173-batched-split-pv-ab-runner-v1", runner)
        self.assertIn('"real_activation_replay_authorized": qualified', runner)
        self.assertIn('"short_tp4_screen_authorized": False', runner)
        self.assertIn('"main_or_yaml_change_authorized": False', runner)


if __name__ == "__main__":
    unittest.main()
