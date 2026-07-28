from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "qwen3_6_scripts" / "corex_moe_direct_routed.cu"
MODEL = ROOT / "qwen3_6_scripts" / "qwen3_5.py"
BUILD = ROOT / "qwen3_6_scripts" / "build_corex_moe_direct_routed.sh"
RUN_CONFIG = ROOT / "computility-run.yaml"


class M198CompensatedW13IntegrationUnitTest(unittest.TestCase):

    def test_production_extension_exposes_both_w13_algorithms(self) -> None:
        source = KERNEL.read_text(encoding="utf-8")
        self.assertIn('module.def("w13", &direct_w13', source)
        self.assertIn(
            'module.def("w13_compensated", &compensated_w13',
            source,
        )
        compensated = source[
            source.index("__global__ void compensated_w13_kernel"):
            source.index("__global__ void direct_w2_reduce_kernel")
        ]
        for fragment in (
            "__fmul_rn",
            "compensated_add",
            "warp_sum_rn",
            "__float2half_rn",
        ):
            self.assertIn(fragment, compensated)
        for fragment in (
            "__fadd_rn",
            "subtract_rn",
            "compensated_add",
            "warp_sum_rn",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("fmaf(", compensated)

    def test_candidate_switch_is_default_off_and_fails_closed(self) -> None:
        source = MODEL.read_text(encoding="utf-8")
        self.assertIn("BI100_MOE_COREX_COMPENSATED_W13", source)
        self.assertIn(
            "_REQUEST_COREX_MOE_COMPENSATED_W13 = env_bool(",
            source,
        )
        self.assertIn(
            '"BI100_MOE_COREX_COMPENSATED_W13 requires "',
            source,
        )
        self.assertIn('"BI100_MOE_COREX_DIRECT_ROUTED=1"', source)
        self.assertIn(
            'hasattr(_corex_moe_direct_routed, "w13_compensated")',
            source,
        )
        self.assertIn(
            "_corex_moe_direct_routed.w13_compensated",
            source,
        )

    def test_candidate_is_not_enabled_by_submission_yaml(self) -> None:
        run_config = RUN_CONFIG.read_text(encoding="utf-8")
        self.assertNotIn("BI100_MOE_COREX_COMPENSATED_W13", run_config)

    def test_existing_build_script_compiles_the_combined_abi(self) -> None:
        source = BUILD.read_text(encoding="utf-8")
        self.assertIn("corex_moe_direct_routed.cu", source)
        self.assertIn(
            "-DTORCH_EXTENSION_NAME=corex_moe_direct_routed",
            source,
        )
        self.assertIn("corex_moe_direct_routed.so", source)


if __name__ == "__main__":
    unittest.main()
