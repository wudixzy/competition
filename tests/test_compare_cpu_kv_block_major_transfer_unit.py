from __future__ import annotations

import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import compare_cpu_kv_block_major_transfer as comparator


def _evidence(candidate_ms: float = 20.0) -> tuple[dict, dict]:
    shape = {
        "block_size": 16,
        "bytes_per_block_per_rank": 163840,
        "dtype": "float16",
        "head_size": 256,
        "local_num_kv_heads": 1,
        "num_attention_layers": 10,
    }
    paged = {
        "schema": comparator.PAGED_SCHEMA,
        "mode": "gate",
        "decision": {"qualified": True},
        "shape": shape,
        "device_name": "Iluvatar BI-V100",
        "torch_version": "2.1.0",
        "results": {},
    }
    candidate = {
        "schema": comparator.BLOCK_MAJOR_SCHEMA,
        "version": 1,
        "decision": {"diagnostic_passed": True},
        "reordered_mapping_exact": True,
        "reordered_mapping_blocks": 513,
        "worker_d2h_before_h2d": True,
        "shape": shape,
        "device_name": "Iluvatar BI-V100",
        "torch_version": "2.1.0",
        "results": {},
    }
    for token_count in comparator.TOKEN_COUNTS:
        bytes_per_direction = token_count * 10240
        paged["results"][str(token_count)] = {
            "bytes_per_direction": bytes_per_direction,
            "d2h_median_ms": 100.0,
            "exact": True,
            "h2d_median_ms": 120.0,
        }
        candidate["results"][str(token_count)] = {
            "bytes_per_direction": bytes_per_direction,
            "d2h_median_ms": candidate_ms,
            "exact": True,
            "h2d_median_ms": candidate_ms,
        }
    return paged, candidate


class BlockMajorComparatorTest(unittest.TestCase):

    def test_qualified_pair(self):
        report = comparator.compare(*_evidence())
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["reasons"], [])

    def test_speed_failure_is_closed(self):
        report = comparator.compare(*_evidence(candidate_ms=30.0))
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("below 4.0x" in reason
                            for reason in report["reasons"]))

    def test_reordered_boundary_is_required(self):
        paged, candidate = _evidence()
        candidate["reordered_mapping_blocks"] = 512
        report = comparator.compare(paged, candidate)
        self.assertFalse(report["qualified"], report)
        self.assertIn(
            "block-major probe did not cover the 512/513 boundary",
            report["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
