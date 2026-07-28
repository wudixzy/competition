from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "tests" / "bench_m1_101_wmma_qk_high_precision.py"
FROZEN = {
    ROOT / "tests" / "bench_attention_wmma_qk.py":
        "55a4ed735abda6e88f2bbb3f4cc264af1b9629062fb62c9dfc130f683c63895f",
    ROOT / "tests" / "build_corex_attention_wmma_qk_probe.sh":
        "9436cd30428f357addf3bcf90d14618a984d48d08f593ac88db70dc6da688958",
    ROOT / "tests" / "corex_attention_wmma_qk_probe.cu":
        "08a68ffc068c7f5a21796b32b64e2164c03f7c1b0270e19d862e116abdd3c688",
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


class M1101WmmaQkOracleUnitTest(unittest.TestCase):

    def test_frozen_m1_28_artifacts_are_byte_exact(self):
        for path, expected in FROZEN.items():
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                expected,
            )

    def test_fixed_shape_and_historical_performance_gate(self):
        values = assignments()
        self.assertEqual(values["TILES"], 128)
        self.assertEqual(values["QUERY_ROWS"], 16)
        self.assertEqual(values["KEY_ROWS"], 32)
        self.assertEqual(values["HEAD_DIM"], 256)
        self.assertEqual(values["MAGNITUDES"], (0.5, 1.0, 2.0))
        self.assertEqual(values["SEED"], 20260718)
        self.assertEqual(values["TIMING_SEED_OFFSET"], 100)
        self.assertEqual(values["WARMUP"], 5)
        self.assertEqual(values["REPEATS"], 20)
        self.assertEqual(values["ORACLE_CPU_THREADS"], 8)
        self.assertEqual(values["MINIMUM_QK_SPEEDUP"], 1.5)
        self.assertEqual(
            values["NONINFERIOR_RELATIVE_L2_SLACK"], 1e-8)

    def test_no_runtime_tuning_surface(self):
        source = ORACLE.read_text(encoding="utf-8")
        for required in (
            'parser.add_argument("--extension"',
            'parser.add_argument("--device"',
            'parser.add_argument("--out"',
        ):
            self.assertIn(required, source)
        for forbidden in (
            '"--tiles"',
            '"--magnitude"',
            '"--seed"',
            '"--warmup"',
            '"--repeats"',
            '"--speedup-gate"',
            '"--relative-l2-gate"',
            '"--max-abs-gate"',
        ):
            self.assertNotIn(forbidden, source)

    def test_policy_v2_noninferiority_and_decision_boundary(self):
        source = ORACLE.read_text(encoding="utf-8")
        for marker in (
            "CPU FP64 QK, softmax, and PV rounded once to FP16",
            "candidate aggregate relative L2 is worse",
            "candidate maximum row relative L2 is worse",
            "candidate maximum absolute error is worse",
            "candidate rounded-oracle mismatch count is worse",
            '"integration_benefit_gate_authorized": qualified',
            '"service_integration_authorized": False',
            '"production_promotion_authorized": False',
            '"yaml_change_authorized": False',
            '"main_merge_authorized": False',
        ):
            self.assertIn(marker, source)

    def test_timing_is_paired_and_order_balanced(self):
        source = ORACLE.read_text(encoding="utf-8")
        self.assertIn("def measure_pair(", source)
        self.assertIn("candidate_first = bool(trial % 2)", source)
        self.assertIn('"paired_order": order', source)

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
