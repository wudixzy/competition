import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import bench_cpu_kv_offload_transfer as benchmark


def passing_case(token_count: int) -> dict[str, object]:
    return {
        "token_count": token_count,
        "exact": True,
        "d2h_median_ms": 100.0,
        "h2d_median_ms": 100.0,
    }


class CpuKvOffloadTransferUnitTest(unittest.TestCase):

    def test_production_block_size_is_160_kib_per_rank(self):
        self.assertEqual(benchmark.bytes_per_block_per_rank(), 163_840)
        self.assertEqual(
            benchmark.bytes_for_tokens(65_536),
            671_088_640,
        )
        with self.assertRaises(ValueError):
            benchmark.blocks_for_tokens(65_535)

    def test_gate_requires_every_fixed_case_and_exact_transfers(self):
        results = {
            str(tokens): passing_case(tokens)
            for tokens in benchmark.GATE_TOKEN_COUNTS
        }
        self.assertTrue(
            benchmark.evaluate_gate("gate", results)["qualified"])

        del results["16384"]
        decision = benchmark.evaluate_gate("gate", results)
        self.assertFalse(decision["qualified"])
        self.assertIn("missing case 16384", decision["reasons"])

        results["16384"] = passing_case(16_384)
        results["65536"]["exact"] = False
        decision = benchmark.evaluate_gate("gate", results)
        self.assertFalse(decision["qualified"])
        self.assertIn("case 65536 is not exact", decision["reasons"])

    def test_gate_enforces_long_transfer_latency_limits(self):
        results = {
            str(tokens): passing_case(tokens)
            for tokens in benchmark.GATE_TOKEN_COUNTS
        }
        results["65536"]["d2h_median_ms"] = 2000.001
        results["131072"]["d2h_median_ms"] = 2600.0
        results["131072"]["h2d_median_ms"] = 2500.0
        decision = benchmark.evaluate_gate("gate", results)
        self.assertFalse(decision["qualified"])
        self.assertEqual(len(decision["reasons"]), 2)

    def test_smoke_can_pass_but_never_qualifies_candidate(self):
        results = {"4096": passing_case(4096)}
        decision = benchmark.evaluate_gate("smoke", results)
        self.assertTrue(decision["smoke_passed"])
        self.assertFalse(decision["qualified"])

    def test_bandwidth_rejects_invalid_time(self):
        self.assertAlmostEqual(
            benchmark.transfer_gib_per_second(1024**3, 1000.0), 1.0)
        with self.assertRaises(ValueError):
            benchmark.transfer_gib_per_second(1024, 0.0)


if __name__ == "__main__":
    unittest.main()
