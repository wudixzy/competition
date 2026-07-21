from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import bench_fused_paged_prefill_attention as benchmark


def _cases(speedup: float = 1.6) -> dict:
    result = {}
    for name, ctx_len, q_len, performance_case in benchmark.CASES:
        case = {
            "ctx_len": ctx_len,
            "finite": True,
            "lse_relative_l2": 5e-6,
            "output_max_abs": 5e-4,
            "output_relative_l2": 5e-6,
            "physical_block_permutation": True,
            "q_len": q_len,
            "total_kv_len": ctx_len + q_len,
        }
        if performance_case:
            candidate_median_ms = 10.0
            reference_median_ms = candidate_median_ms * speedup
            case.update({
                "candidate_median_ms": candidate_median_ms,
                "candidate_trials_ms": [candidate_median_ms] * 7,
                "reference_median_ms": reference_median_ms,
                "reference_trials_ms": [reference_median_ms] * 7,
                "speedup": speedup,
            })
        result[name] = case
    return result


class FusedPagedPrefillGateTest(unittest.TestCase):

    def test_default_extension_loads_from_vllm(self):
        sentinel = object()
        with mock.patch.object(
                benchmark.importlib, "import_module",
                return_value=sentinel) as import_module:
            self.assertIs(benchmark._load_extension(None), sentinel)
        import_module.assert_called_once_with(
            "vllm.corex_fused_paged_prefill")

    def test_isolated_extension_uses_compiled_module_name(self):
        sentinel = object()
        loader = mock.Mock()
        spec = mock.Mock(loader=loader)
        with tempfile.TemporaryDirectory() as directory:
            extension_path = pathlib.Path(directory) / "candidate.so"
            extension_path.touch()
            with mock.patch.object(
                    benchmark.importlib.util, "spec_from_file_location",
                    return_value=spec) as make_spec, mock.patch.object(
                        benchmark.importlib.util, "module_from_spec",
                        return_value=sentinel):
                self.assertIs(
                    benchmark._load_extension(extension_path), sentinel)
        make_spec.assert_called_once_with(
            "corex_fused_paged_prefill", extension_path.resolve())
        loader.exec_module.assert_called_once_with(sentinel)

    def test_fixed_geometry(self):
        self.assertEqual(benchmark.BLOCK_SIZE, 16)
        self.assertEqual(benchmark.BLOCKS_PER_TILE, 32)
        self.assertEqual(benchmark.NUM_Q_HEADS, 6)
        self.assertEqual(benchmark.NUM_KV_HEADS, 1)
        self.assertEqual(benchmark.HEAD_DIM, 256)
        self.assertEqual(benchmark.WARMUP_TRIALS, 5)
        self.assertEqual(benchmark.MEASURED_TRIALS, 7)
        self.assertIn(
            ("service_65k_q8192", 65_536, 8_192, False), benchmark.CASES)

    def test_qualified_report(self):
        report = benchmark.evaluate(_cases())
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["reasons"], [])

    def test_numerical_failure_is_closed(self):
        cases = _cases()
        cases["paged_65520_q16"]["output_relative_l2"] = 2e-5
        report = benchmark.evaluate(cases)
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("output_relative_l2" in reason
                            for reason in report["reasons"]))

    def test_wrong_geometry_is_rejected(self):
        cases = _cases()
        cases["perf_235k"]["ctx_len"] = 234_720
        report = benchmark.evaluate(cases)
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("ctx_len must equal 234736" in reason
                            for reason in report["reasons"]))

    def test_identity_physical_layout_is_rejected(self):
        cases = _cases()
        cases["paged_65520_q16"]["physical_block_permutation"] = False
        report = benchmark.evaluate(cases)
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("non-identity block table" in reason
                            for reason in report["reasons"]))

    def test_performance_failure_is_closed(self):
        report = benchmark.evaluate(_cases(speedup=1.49))
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("below 1.5x" in reason
                            for reason in report["reasons"]))

    def test_incomplete_trials_are_rejected(self):
        cases = _cases()
        cases["perf_74k"]["candidate_trials_ms"] = [10.0] * 6
        report = benchmark.evaluate(cases)
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("7 positive candidate_trials_ms" in reason
                            for reason in report["reasons"]))

    def test_inconsistent_speedup_is_rejected(self):
        cases = _cases()
        cases["perf_128k"]["speedup"] = 1.7
        report = benchmark.evaluate(cases)
        self.assertFalse(report["qualified"], report)
        self.assertTrue(any("speedup does not match medians" in reason
                            for reason in report["reasons"]))


if __name__ == "__main__":
    unittest.main()
