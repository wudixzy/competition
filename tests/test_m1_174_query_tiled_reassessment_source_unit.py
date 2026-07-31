from __future__ import annotations

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "qwen3_6_scripts"
    / "corex_query_tiled_paged_prefill_reassessed.cu")
BUILD = (
    ROOT / "qwen3_6_scripts"
    / "build_corex_query_tiled_paged_prefill_reassessed.sh")
RUNNER = ROOT / "scripts" / "run_m1_174_query_tiled_reassessment.py"
PATCH_OPS = ROOT / "qwen3_6_scripts" / "patch_ops.sh"


class M1174QueryTiledReassessmentSourceTest(unittest.TestCase):
    def test_source_is_exact_historical_single_group_design(self):
        payload = SOURCE.read_bytes()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "0217061a8803d2a181a01dd7316531d8cfed1fb84619d5f4e204acafe53b89c5",
        )
        source = payload.decode("utf-8")
        self.assertNotIn("kPvReductionSplits", source)
        self.assertNotIn("partial_output", source)
        self.assertIn("query_tiled_paged_prefill_kernel", source)
        self.assertGreaterEqual(source.count("wmma::mma_sync("), 2)
        self.assertIn("float running_output[kQueryTile * kHeadDim]", source)
        self.assertNotIn("cublas", source.lower())
        self.assertNotIn("split_output", source)

    def test_build_and_runner_are_isolated_and_fail_closed(self):
        build = BUILD.read_text(encoding="utf-8")
        self.assertIn("--cuda-gpu-arch=ivcore10", build)
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_query_tiled_paged_prefill_reassessed",
            build,
        )
        self.assertNotIn(
            "corex_query_tiled_paged_prefill_reassessed",
            PATCH_OPS.read_text(encoding="utf-8"),
        )
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn("bench_m1_162_calibrated_fp16_qk_ab.py", runner)
        self.assertIn("bi100-m1-174-query-tiled-reassessment-runner-v1", runner)
        self.assertIn('"real_activation_replay_authorized": qualified', runner)
        self.assertIn('"short_tp4_screen_authorized": False', runner)
        self.assertIn('"main_or_yaml_change_authorized": False', runner)


if __name__ == "__main__":
    unittest.main()
