import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class GdnRecurrentExtensionStaticTests(unittest.TestCase):
    def test_build_does_not_require_ninja(self):
        setup_source = (
            ROOT / "qwen3_6_scripts" / "bi100_ext" / "setup.py"
        ).read_text(encoding="utf-8")
        self.assertIn("BuildExtension.with_options(use_ninja=False)",
                      setup_source)

    def test_patch_builds_and_import_checks_extension(self):
        patch_source = (
            ROOT / "qwen3_6_scripts" / "patch_ops.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--no-build-isolation", patch_source)
        self.assertIn("import bi100_gdn_recurrent", patch_source)
        self.assertIn("recurrent_update", patch_source)

    def test_decode_keeps_runtime_fallback(self):
        model_source = (
            ROOT / "qwen3_6_scripts" / "qwen3_5.py"
        ).read_text(encoding="utf-8")
        self.assertIn('env_bool("BI100_GDN_RECURRENT_EXT", True)',
                      model_source)
        self.assertIn("_bi100_gdn_recurrent.recurrent_update", model_source)
        self.assertIn("ts_flat.baddbmm_", model_source)
        self.assertIn("temporal_state.mul_", model_source)

    def test_kernel_preserves_float32_state_contract(self):
        kernel_source = (
            ROOT / "qwen3_6_scripts" / "bi100_ext" / "gdn_recurrent.cu"
        ).read_text(encoding="utf-8")
        self.assertIn("tensor.scalar_type() == at::kFloat", kernel_source)
        self.assertIn("state must have shape (B, H, 128, 128)", kernel_source)
        self.assertIn("getCurrentCUDAStream", kernel_source)


if __name__ == "__main__":
    unittest.main()
