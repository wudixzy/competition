from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m1_111_query_tiled_production_retest.sh"
BUILD = ROOT / "tests" / "build_m1_111_query_tiled_retest.sh"
SOURCE = (
    ROOT / "tests" / "corex_query_tiled_paged_prefill_m1_55_a30b6e7.cu")
BENCH = ROOT / "tests" / "bench_m1_111_query_tiled_production_retest.py"


def load_benchmark_module():
    sys.path.insert(0, str(ROOT / "tests"))
    spec = importlib.util.spec_from_file_location(
        "m1_111_retest_unit", BENCH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M1-111 benchmark module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class M1111QueryTiledProductionRetestUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.bench_source = BENCH.read_text(encoding="utf-8")
        cls.bench = load_benchmark_module()

    def test_candidate_source_matches_the_best_m1_55_revision(self) -> None:
        expected = subprocess.run(
            [
                "git",
                "rev-parse",
                "a30b6e7:qwen3_6_scripts/"
                "corex_query_tiled_paged_prefill.cu",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        actual = subprocess.run(
            ["git", "hash-object", str(SOURCE)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(actual, expected)
        self.assertIn(
            "a30b6e7212286cd613c946b1ca02d8972a198863",
            self.bench_source,
        )

    def test_fixed_domain_covers_production_and_capacity_shapes(self) -> None:
        self.assertEqual(
            self.bench.CASES["boundary_262k_q8192"][:2],
            (253_952, 8_192),
        )
        self.assertEqual(
            sum(self.bench.CASES["boundary_262k_q8192"][:2]),
            262_144,
        )
        self.assertTrue(
            all(query_len >= 4_096 for _, query_len, _ in
                self.bench.CASES.values())
        )
        for marker in (
            "production_dense_q8176",
            "production_32k_q8176",
            "production_65k_q8176",
            "production_128k_q8176",
            "production_235k_q5616",
            "boundary_262k_q8192",
        ):
            self.assertIn(marker, self.runner)

    def test_numerical_gate_is_fixed_and_fail_closed(self) -> None:
        passing = {
            "finite": True,
            "output_relative_l2": 1e-5,
            "lse_relative_l2": 1e-5,
            "output_max_abs": 1e-3,
        }
        self.assertEqual(
            self.bench.numerical_reasons("candidate", passing), [])
        failing = dict(passing)
        failing["output_relative_l2"] = 1.0001e-5
        self.assertEqual(
            len(self.bench.numerical_reasons("candidate", failing)), 1)
        self.assertIn("MIN_LONG_SPEEDUP = 1.5", self.bench_source)
        self.assertIn(
            '"tp4_service_authorized": False',
            self.bench_source,
        )
        self.assertIn(
            '"main_or_yaml_change_authorized": False',
            self.bench_source,
        )

    def test_build_uses_a_distinct_module_without_changing_source(self) -> None:
        self.assertIn(
            "-DTORCH_EXTENSION_NAME="
            "corex_query_tiled_paged_prefill_retest",
            self.build,
        )
        self.assertIn(
            "corex_query_tiled_paged_prefill_retest.so",
            self.build,
        )
        self.assertIn("constexpr int kWarpSize = 64;", self.source)
        self.assertNotIn("computility-run.yaml", self.build)

    def test_runner_uses_four_gpus_and_scoped_cleanup(self) -> None:
        for marker in (
            "for gpu in 0 1 2 3",
            'CUDA_VISIBLE_DEVICES="$GPU"',
            "setsid",
            "bi100_stop_process_group",
            '"$pid" "$pid" 60 20',
            "trap finish EXIT",
            "timeout --foreground --signal=TERM --kill-after=60s",
            "service_postflight_gate.py",
            "bi100_preflight.py",
            "compare_bi100_preflights.py",
            "fatal_scan.rc",
            '"unchanged_m1_55_route_closed"',
        ):
            self.assertIn(marker, self.runner)
        for forbidden in ("pkill", "killall", "git push"):
            self.assertNotIn(forbidden, self.runner)

    def test_invalid_invocation_fails_before_runtime_access(self) -> None:
        result = subprocess.run(
            ["bash", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
