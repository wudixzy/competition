from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "vllm" / "worker" / "bi100_block_major_kv.py"
SPEC = importlib.util.spec_from_file_location("bi100_block_major_kv", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
block_major = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(block_major)


class BlockMajorKvHelpersTest(unittest.TestCase):

    def test_layout_selector_defaults_and_validates(self):
        self.assertEqual(block_major.transfer_layout_from_env({}), "paged")
        self.assertEqual(
            block_major.transfer_layout_from_env({
                "BI100_CPU_KV_TRANSFER_LAYOUT": "block_major",
            }),
            "block_major",
        )
        with self.assertRaises(RuntimeError):
            block_major.transfer_layout_from_env({
                "BI100_CPU_KV_TRANSFER_LAYOUT": "auto",
            })

    def test_fixed_chunk_boundaries(self):
        self.assertEqual(block_major.chunk_ranges(0), ())
        self.assertEqual(block_major.chunk_ranges(1), ((0, 1),))
        self.assertEqual(block_major.chunk_ranges(512), ((0, 512),))
        self.assertEqual(
            block_major.chunk_ranges(513), ((0, 512), (512, 513)))
        self.assertEqual(
            block_major.chunk_ranges(1025),
            ((0, 512), (512, 1024), (1024, 1025)),
        )

    def test_contiguous_run_detection(self):
        self.assertIsNone(block_major.contiguous_start(()))
        self.assertEqual(block_major.contiguous_start((7,)), 7)
        self.assertEqual(block_major.contiguous_start((7, 8, 9)), 7)
        self.assertIsNone(block_major.contiguous_start((7, 9)))
        with self.assertRaises(TypeError):
            block_major.contiguous_start((7, True))

    def test_mapping_accepts_reordered_unique_pairs(self):
        self.assertEqual(
            block_major.validate_mapping_pairs(
                [(3, 5), (1, 2), (7, 0)],
                source_limit=8,
                destination_limit=6,
            ),
            ((3, 5), (1, 2), (7, 0)),
        )

    def test_mapping_rejects_duplicates_and_bounds(self):
        with self.assertRaisesRegex(ValueError, "duplicate mapping source"):
            block_major.validate_mapping_pairs(
                [(1, 2), (1, 3)], 8, 8)
        with self.assertRaisesRegex(ValueError, "duplicate mapping destination"):
            block_major.validate_mapping_pairs(
                [(1, 2), (3, 2)], 8, 8)
        with self.assertRaisesRegex(ValueError, "outside"):
            block_major.validate_mapping_pairs([(8, 0)], 8, 8)
        with self.assertRaisesRegex(ValueError, "outside"):
            block_major.validate_mapping_pairs([(0, 8)], 8, 8)

    def test_mapping_rejects_malformed_rows(self):
        with self.assertRaises(ValueError):
            block_major.validate_mapping_pairs([(1,)], 8, 8)
        with self.assertRaises(TypeError):
            block_major.validate_mapping_pairs([(True, 1)], 8, 8)


if __name__ == "__main__":
    unittest.main()
