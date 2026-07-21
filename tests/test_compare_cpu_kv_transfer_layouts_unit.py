import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/compare_cpu_kv_transfer_layouts.py"
SPEC = importlib.util.spec_from_file_location("transfer_layouts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


SHAPE = {
    "num_attention_layers": 10,
    "block_size": 16,
    "local_num_kv_heads": 1,
    "head_size": 256,
    "dtype": "float16",
    "bytes_per_block_per_rank": 163840,
}


def _report(layout, speedup=4.0):
    results = {}
    for tokens in MODULE.TOKEN_COUNTS:
        contiguous_ms = 10.0
        elapsed = contiguous_ms * speedup if layout == "paged" else contiguous_ms
        results[str(tokens)] = {
            "exact": True,
            "bytes_per_direction": tokens * 10_240,
            "d2h_median_ms": elapsed,
            "h2d_median_ms": elapsed,
        }
    if layout == "paged":
        return {
            "schema": MODULE.PAGED_SCHEMA,
            "mode": "gate",
            "shape": SHAPE,
            "device_name": "BI100",
            "torch_version": "2.1",
            "results": results,
            "decision": {"qualified": True},
        }
    return {
        "schema": MODULE.CONTIGUOUS_SCHEMA,
        "version": MODULE.VERSION,
        "shape": SHAPE,
        "device_name": "BI100",
        "torch_version": "2.1",
        "results": results,
        "decision": {"diagnostic_passed": True},
    }


class CpuKvTransferLayoutsUnitTest(unittest.TestCase):

    def test_exact_four_x_qualifies(self):
        report = MODULE.compare(_report("paged", 4.0),
                                _report("contiguous"))
        self.assertTrue(report["qualified"])
        self.assertEqual(report["reasons"], [])

    def test_below_threshold_fails(self):
        report = MODULE.compare(_report("paged", 3.999),
                                _report("contiguous"))
        self.assertFalse(report["qualified"])
        self.assertTrue(any("below 4.0x" in reason
                            for reason in report["reasons"]))

    def test_contract_mismatch_and_nonexact_fail(self):
        paged = _report("paged")
        contiguous = _report("contiguous")
        contiguous["shape"] = {**SHAPE, "block_size": 32}
        contiguous["results"]["65536"]["exact"] = False
        report = MODULE.compare(paged, contiguous)
        self.assertFalse(report["qualified"])
        self.assertIn("transfer shapes differ", report["reasons"])
        self.assertTrue(any("not exact" in reason
                            for reason in report["reasons"]))


if __name__ == "__main__":
    unittest.main()
