from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "bench_fused_prefill_service.py"


class FusedPrefillServiceBenchmarkUnitTest(unittest.TestCase):

    def test_benchmark_encodes_cold_warm_and_output_gates(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("for _ in range(3)", source)
        self.assertIn('requests[0]["cached_tokens"] != 0', source)
        self.assertIn("target - 32", source)
        self.assertIn("warm outputs differ", source)
        self.assertIn("cold/warm outputs differ", source)
        self.assertIn('"output_tps_p10"', source)
        self.assertNotIn("content_parts", source.split("report =", 1)[1])

    def test_percentile_is_interpolated(self):
        spec = importlib.util.spec_from_file_location("m1_47_service_bench", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(module._percentile([], 90), 0.0)
        self.assertEqual(module._percentile([2.0], 90), 2.0)
        self.assertAlmostEqual(module._percentile([1.0, 3.0], 10), 1.2)


if __name__ == "__main__":
    unittest.main()
