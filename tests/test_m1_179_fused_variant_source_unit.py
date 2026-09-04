from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "qwen3_6_scripts/build_corex_fused_paged_prefill_variant_runtime.sh"
PAGED = ROOT / "qwen3_6_scripts/paged_attn.py"


class M1179FusedVariantSourceTests(unittest.TestCase):

    def test_build_variants_share_toolchain_and_flags(self) -> None:
        source = BUILD.read_text(encoding="utf-8")
        self.assertIn("m1_109_fp32_qk", source)
        self.assertIn("corex_fused_paged_prefill_split4.cu", source)
        self.assertIn("m1_162_fp16_qk", source)
        self.assertIn("corex_fused_paged_prefill_fp16_qk.cu", source)
        self.assertEqual(source.count('"${COREX_ROOT}/bin/clang++"'), 1)
        self.assertIn("-DTORCH_EXTENSION_NAME=corex_fused_paged_prefill", source)

    def test_dispatch_marker_includes_variant_and_loaded_module(self) -> None:
        source = PAGED.read_text(encoding="utf-8")
        self.assertIn("BI100_ATTN_COREX_FUSED_PREFILL_VARIANT", source)
        self.assertIn('f"variant={_FUSED_PREFILL_VARIANT} "', source)
        self.assertIn("getattr(_corex_fused_paged_prefill, '__file__', '?')",
                      source)


if __name__ == "__main__":
    unittest.main()
