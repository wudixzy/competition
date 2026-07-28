from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests" / "bench_prefix_cold_chunk_high_precision.py"
FROZEN = {
    ROOT / "tests" / "bench_prefix_attention_breakdown.py":
        "2ab82f69e7833dc2965b03e4cbcebe5beafd9d4954a3e3babda101bb54a0ddd2",
    ROOT / "tests" / "bench_prefix_cold_chunk_hybrid.py":
        "e2dffa151c99f4cf28d827877db68bbcb0a0c0bd6433c466017c255df2f3d076",
}


def assignments() -> dict[str, object]:
    tree = ast.parse(ORACLE.read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            result[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    return result


class M1100PrefixOracleUnitTest(unittest.TestCase):

    def test_frozen_e_prefix_artifacts_are_byte_exact(self):
        for path, expected in FROZEN.items():
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                expected,
            )

    def test_oracle_uses_fixed_production_shape_and_old_thresholds(self):
        values = assignments()
        self.assertEqual(values["PRODUCTION_QUERY_LEN"], 8176)
        self.assertEqual(values["QUERY_HEADS"], 4)
        self.assertEqual(values["KV_HEADS"], 1)
        self.assertEqual(values["HEAD_DIM"], 256)
        self.assertEqual(values["BLOCK_SIZE"], 16)
        self.assertEqual(values["TILE_SIZE"], 512)
        self.assertEqual(values["PRIMARY_CONTEXT"], 65536)
        self.assertEqual(values["PARTIAL_CONTEXT"], 65552)
        self.assertEqual(values["SEEDS"], (20260716, 20260727))
        self.assertEqual(values["WARMUP"], 1)
        self.assertEqual(values["REPEATS"], 3)
        self.assertEqual(values["ORACLE_CPU_THREADS"], 8)
        self.assertEqual(values["PARTIAL_SEED_OFFSET"], 100)
        self.assertEqual(values["MIN_PRIMARY_REDUCTION"], 0.15)
        self.assertEqual(
            values["NONINFERIOR_RELATIVE_L2_SLACK"], 1e-8)

    def test_oracle_has_no_runtime_tuning_surface(self):
        source = ORACLE.read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--device"', source)
        self.assertIn('parser.add_argument("--out"', source)
        for forbidden in (
            '"--tile-size"',
            '"--query-len"',
            '"--query-heads"',
            '"--dtype"',
            '"--relative-l2-gate"',
            '"--max-abs-gate"',
            '"--seed"',
            '"--repeats"',
        ):
            self.assertNotIn(forbidden, source)

    def test_oracle_implements_policy_v2_noninferiority(self):
        source = ORACLE.read_text(encoding="utf-8")
        for marker in (
            "CPU FP64 sampled full-sequence attention rounded once to FP16",
            "candidate aggregate relative L2 is worse",
            "candidate maximum step relative L2 is worse",
            "candidate maximum absolute error is worse",
            "candidate rounded-oracle mismatch count is worse",
            '"next_token_gate_authorized": not reasons',
            '"service_integration_authorized": False',
            '"production_promotion_authorized": False',
            '"yaml_change_authorized": False',
            '"main_merge_authorized": False',
        ):
            self.assertIn(marker, source)

    def test_timing_is_paired_and_order_balanced(self):
        source = ORACLE.read_text(encoding="utf-8")
        self.assertIn("def measure_pair(", source)
        self.assertIn("candidate_first=bool(seed % 2)", source)
        self.assertIn('"paired_order": measured_order', source)

    def test_report_does_not_retain_raw_tensors(self):
        source = ORACLE.read_text(encoding="utf-8")
        report = source.split("report = {", 1)[1]
        for forbidden in (
            '"query":',
            '"key":',
            '"value":',
            '"output":',
            ".tolist()",
        ):
            self.assertNotIn(forbidden, report)


if __name__ == "__main__":
    unittest.main()
