from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "tests" / "bench_m1_56_block_major_kv_transfer.py"
SOURCE = ROOT / "qwen3_6_scripts" / "corex_block_major_kv_transfer.cu"
BUILD = ROOT / "tests" / "build_corex_block_major_kv_transfer.sh"
PATCH_OPS = (ROOT / "qwen3_6_scripts" / "patch_ops.sh").read_text()


def load_benchmark():
    spec = importlib.util.spec_from_file_location("m1_56_benchmark", BENCHMARK)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M1-56 benchmark")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_benchmark()


def qualified_report():
    cases = {}
    for token_count in (
            MODULE.BOUNDARY_TOKEN_COUNTS + MODULE.GATE_TOKEN_COUNTS):
        cases[str(token_count)] = {
            "d2h_exact": True,
            "h2d_exact": True,
            "same_gpu_slot_order_exact": True,
            "d2h_speedup": 4.25,
            "h2d_speedup": 5.5,
            "candidate_components_ms": {
                "d2h_pack": 1.0,
                "d2h_dma": 2.0,
                "d2h_cpu_scatter": 3.0,
                "h2d_cpu_gather": 3.0,
                "h2d_dma": 2.0,
                "h2d_gpu_scatter": 1.0,
            },
        }
    return {
        "mode": "gate",
        "extension_isolated_from_runtime": True,
        "cases": cases,
    }


class M156BlockMajorTransferUnitTests(unittest.TestCase):

    def test_fixed_geometry_matches_qwen_tp4_rank(self):
        self.assertEqual(MODULE.blocks_for_tokens(65536), 4096)
        self.assertEqual(MODULE.blocks_for_tokens(131072), 8192)
        self.assertEqual(MODULE.bytes_per_block_per_rank(), 163840)
        self.assertEqual(MODULE.bytes_for_tokens(65536), 671088640)
        self.assertEqual(MODULE.STAGING_BLOCKS, 512)

    def test_mapping_chunks_cover_512_and_513_boundaries(self):
        chunks = MODULE.mapping_chunks(list(range(512)), list(range(512)))
        self.assertEqual([len(source) for source, _ in chunks], [512])
        chunks = MODULE.mapping_chunks(list(range(513)), list(range(513)))
        self.assertEqual([len(source) for source, _ in chunks], [512, 1])

    def test_mapping_validation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            MODULE.mapping_chunks([0, 0], [1, 2])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            MODULE.mapping_chunks([0, 1], [2, 2])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MODULE.mapping_chunks([-1], [0])
        with self.assertRaisesRegex(ValueError, "differ"):
            MODULE.mapping_chunks([0], [0, 1])

    def test_complete_gate_qualifies(self):
        result = MODULE.evaluate_gate(qualified_report())
        self.assertTrue(result["qualified"], json.dumps(result))

    def test_speedup_and_exactness_fail_closed(self):
        report = qualified_report()
        report["cases"]["65536"]["d2h_speedup"] = 3.999
        report["cases"]["131072"]["h2d_exact"] = False
        result = MODULE.evaluate_gate(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("3.999" in reason for reason in result["reasons"]))
        self.assertTrue(any("H2D is not byte-exact"
                            in reason for reason in result["reasons"]))

    def test_smoke_cannot_qualify(self):
        report = qualified_report()
        report["mode"] = "smoke"
        report["cases"] = {
            key: value for key, value in report["cases"].items()
            if int(key) in MODULE.BOUNDARY_TOKEN_COUNTS
        }
        result = MODULE.evaluate_gate(report)
        self.assertFalse(result["qualified"])
        self.assertTrue(result["smoke_passed"])

    def test_extension_is_strict_and_experimental(self):
        source = SOURCE.read_text()
        self.assertIn("expected exactly ", source)
        self.assertIn("must have shape [2, blocks, 4096]", source)
        self.assertIn("CPU block id out of range", source)
        self.assertIn("C10_CUDA_KERNEL_LAUNCH_CHECK", source)
        self.assertIn("kAttentionLayers = 10", source)
        self.assertIn("kElementsPerPlaneBlock = 4096", source)
        self.assertNotIn("corex_block_major_kv_transfer", PATCH_OPS)

    def test_build_is_fixed_to_corex_ivcore10(self):
        source = BUILD.read_text()
        self.assertIn("--cuda-gpu-arch=ivcore10", source)
        self.assertIn("corex_block_major_kv_transfer.cu", source)
        self.assertIn("corex_block_major_kv_transfer.so", source)


if __name__ == "__main__":
    unittest.main()
