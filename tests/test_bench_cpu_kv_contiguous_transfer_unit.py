import importlib.util
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/bench_cpu_kv_contiguous_transfer.py"
SPEC = importlib.util.spec_from_file_location("contiguous_transfer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _case(exact=True, d2h=10.0, h2d=11.0):
    return {
        "exact": exact,
        "d2h_median_ms": d2h,
        "h2d_median_ms": h2d,
    }


class CpuKvContiguousTransferUnitTest(unittest.TestCase):

    def test_fixed_shape_matches_paged_evidence(self):
        self.assertEqual(MODULE.bytes_per_block_per_rank(), 163_840)
        self.assertEqual(MODULE.bytes_for_tokens(65_536), 671_088_640)
        self.assertEqual(MODULE.bytes_for_tokens(131_072), 1_342_177_280)
        with self.assertRaises(ValueError):
            MODULE.bytes_for_tokens(65_535)

    def test_diagnostic_requires_all_exact_finite_cases(self):
        results = {str(tokens): _case() for tokens in MODULE.TOKEN_COUNTS}
        self.assertTrue(MODULE.evaluate(results)["diagnostic_passed"])
        self.assertFalse(MODULE.evaluate(results)["qualified"])

        del results["65536"]
        self.assertFalse(MODULE.evaluate(results)["diagnostic_passed"])
        results["65536"] = _case(exact=False)
        self.assertFalse(MODULE.evaluate(results)["diagnostic_passed"])
        results["65536"] = _case(d2h=math.inf)
        self.assertFalse(MODULE.evaluate(results)["diagnostic_passed"])

    def test_bandwidth_rejects_nonpositive_time(self):
        self.assertAlmostEqual(MODULE.gib_per_second(1024**3, 1000), 1.0)
        with self.assertRaises(ValueError):
            MODULE.gib_per_second(1024, 0)


if __name__ == "__main__":
    unittest.main()
