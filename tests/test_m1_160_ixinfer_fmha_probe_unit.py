from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "tests" / "corex_ixinfer_fmha_probe_ext.cu"
).read_text(encoding="utf-8")
BUILD = (
    ROOT / "tests" / "build_corex_ixinfer_fmha_probe.sh"
).read_text(encoding="utf-8")
RUNNER_PATH = ROOT / "tests" / "run_corex_ixinfer_fmha_probe.py"
RUNNER = RUNNER_PATH.read_text(encoding="utf-8")
PATCH_OPS = (
    ROOT / "qwen3_6_scripts" / "patch_ops.sh"
).read_text(encoding="utf-8")


class M1160IxinferFmhaProbeTest(unittest.TestCase):
    def test_runner_syntax(self):
        ast.parse(RUNNER, filename=str(RUNNER_PATH))

    def test_probe_covers_production_shape_and_one_control(self):
        self.assertIn("head_size == 128 || head_size == 256", SOURCE)
        self.assertIn(
            '"--head-size", type=int, choices=(128, 256), default=256',
            RUNNER,
        )
        self.assertIn("query_heads % kv_heads == 0", SOURCE)
        self.assertIn("config.kvHeadNum", SOURCE)
        self.assertIn("config.isCausal = causal", SOURCE)
        self.assertIn("CUINFER_FATTN_BSHD", SOURCE)
        self.assertIn("cuinferFMHAForwardEx(", SOURCE)
        self.assertIn("query.stride(sequence_dimension)", SOURCE)
        self.assertIn("key.stride(sequence_dimension)", SOURCE)
        self.assertIn("value.stride(sequence_dimension)", SOURCE)

    def test_probe_checks_every_cuinfer_call_and_uses_current_stream(self):
        for operation in (
            "cuinferCreate",
            "cuinferCreateTensorDescriptor",
            "cuinferSetTensor4dDescriptorEx",
            "cuinferSetStream",
            "cuinferFMHAForwardEx",
        ):
            self.assertIn(f'"{operation}"', SOURCE)
        self.assertIn("at::cuda::getCurrentCUDAStream()", SOURCE)

    def test_probe_is_isolated_from_runtime_overlay(self):
        self.assertIn(
            "OUTPUT=${OUTPUT_DIR}/corex_ixinfer_fmha_probe.so", BUILD
        )
        self.assertIn("-lcuinfer", BUILD)
        self.assertNotIn("corex_ixinfer_fmha_probe", PATCH_OPS)
        self.assertIn('"runtime_overlay_authorized": False', RUNNER)
        self.assertIn('"tp4_service_authorized": False', RUNNER)
        self.assertIn('"main_or_yaml_change_authorized": False', RUNNER)

    def test_runner_uses_bottom_right_causal_reference(self):
        self.assertIn(
            "key_length - query_length + query_positions", RUNNER
        )
        self.assertIn(
            "key_float.repeat_interleave(repeats, dim=2)", RUNNER
        )
        self.assertIn(
            "value_float.repeat_interleave(repeats, dim=2)", RUNNER
        )
        self.assertIn(
            'choices=("bshd", "bhsd")', RUNNER
        )
        self.assertIn(
            "query_bshd.permute(0, 2, 1, 3).contiguous()", RUNNER
        )
        self.assertIn(
            'result["numerical"]["relative_l2"] <= 1e-5', RUNNER
        )


if __name__ == "__main__":
    unittest.main()
