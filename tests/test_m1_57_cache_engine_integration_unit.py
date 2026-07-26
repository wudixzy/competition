import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "bench_m1_57_cache_engine_integration.py"


class M157CacheEngineIntegrationHarnessTest(unittest.TestCase):

    def test_harness_has_fixed_geometry_and_no_tuning_arguments(self):
        text = HARNESS.read_text(encoding="utf-8")
        module = ast.parse(text)
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {
                "NUM_ATTENTION_LAYERS",
                "KV_PLANES",
                "ELEMENTS_PER_PLANE_BLOCK",
                "GPU_BLOCKS",
                "CPU_BLOCKS",
                "FIXED_SEED",
            }
        }
        self.assertEqual(assignments, {
            "NUM_ATTENTION_LAYERS": 10,
            "KV_PLANES": 2,
            "ELEMENTS_PER_PLANE_BLOCK": 4096,
            "GPU_BLOCKS": 1025,
            "CPU_BLOCKS": 1536,
            "FIXED_SEED": 20260726,
        })
        for forbidden in (
                "--gpu-blocks",
                "--cpu-blocks",
                "--staging-blocks",
                "--chunk-blocks",
                "--buffers"):
            self.assertNotIn(forbidden, text)

    def test_harness_uses_patched_cache_engine_and_fail_closed_gates(self):
        text = HARNESS.read_text(encoding="utf-8")
        for required in (
                "engine.swap_out",
                "engine.swap_in",
                "round_trip_byte_exact",
                "same_slot_preserved_victim_exact",
                "same_slot_promoted_request_exact",
                "invalid_mapping_fail_fast",
                "invalid_mapping_zero_write",
                "invalid_selector_fail_fast",
            "default_selector_off"):
            self.assertIn(required, text)
        self.assertIn(
            "7e2aafd8dc755b0ee16c3b9bb812b955"
            "48fc042bbaa840dd9db7d2c51a10474c",
            text,
        )
        self.assertNotIn("temperature", text)
        self.assertNotIn("model.generate", text)


if __name__ == "__main__":
    unittest.main()
