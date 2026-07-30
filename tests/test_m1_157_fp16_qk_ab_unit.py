from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(TESTS), str(SCRIPTS)]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CELL = _load(
    "bench_m1_157_fp16_qk_ab",
    TESTS / "bench_m1_157_fp16_qk_ab.py",
)
RUNNER = _load(
    "run_m1_157_fp16_qk_ab",
    SCRIPTS / "run_m1_157_fp16_qk_ab.py",
)


def _comparison() -> dict:
    return {
        "finite": True,
        "output_max_abs": 2.0e-4,
        "output_relative_l2": 5.0e-6,
        "lse_relative_l2": 2.0e-8,
    }


def report(case: str, gpu: int, speedup: float = 1.10) -> dict:
    context_len, query_len = CELL.CASES[case]
    trials = [10.1, 10.0, 9.9, 10.2, 9.8]
    value = {
        "schema": CELL.SCHEMA,
        "version": 1,
        "source_revision": "b" * 40,
        "case": case,
        "context_len": context_len,
        "query_len": query_len,
        "seed": CELL.production.SEED,
        "warmups": CELL.WARMUPS,
        "trials": CELL.TRIALS,
        "visible_physical_gpu": gpu,
        "baseline_extension": {"sha256": "a" * 64},
        "candidate_extension": {"sha256": "c" * 64},
        "timings": {
            "baseline": {
                "cuda_trials_ms": trials,
                "cuda_median_ms": 10.0,
            },
            "candidate": {
                "cuda_trials_ms": trials,
                "cuda_median_ms": 10.0 / speedup,
            },
            "speedup": speedup,
        },
        "numerical": {
            "baseline_vs_reference": _comparison(),
            "candidate_vs_reference": _comparison(),
            "candidate_vs_baseline": _comparison(),
        },
        "authorization": {
            "operator_screen_only": True,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }
    value["evaluation"] = CELL.evaluate(value)
    return value


class M1157Fp16QkAbTest(unittest.TestCase):
    def test_fixed_grid_targets_p90_region(self):
        self.assertEqual(tuple(CELL.CASES), RUNNER.CASES)
        self.assertEqual(
            [context + query for context, query in CELL.CASES.values()],
            [16_368, 32_752, 65_520],
        )

    def test_cell_fails_closed_on_numeric_error_or_regression(self):
        value = report(RUNNER.CASES[0], 1)
        self.assertTrue(value["evaluation"]["qualified"])
        value["numerical"]["candidate_vs_reference"][
            "output_relative_l2"
        ] = 2.0e-5
        self.assertFalse(CELL.evaluate(value)["qualified"])
        value = report(RUNNER.CASES[0], 1, speedup=0.97)
        self.assertFalse(value["evaluation"]["qualified"])

    def test_aggregate_requires_eight_percent_median_gain(self):
        reports = [
            report(case, gpu, speedup)
            for case, gpu, speedup in zip(
                RUNNER.CASES, (1, 2, 3), (1.06, 1.10, 1.12)
            )
        ]
        value = RUNNER.aggregate(
            reports,
            revision="b" * 40,
            baseline_sha="a" * 64,
            candidate_sha="c" * 64,
        )
        self.assertTrue(value["qualified"], value)
        self.assertTrue(
            value["authorization"]["short_tp4_screen_authorized"]
        )
        reports[1]["timings"]["speedup"] = 1.07
        value = RUNNER.aggregate(
            reports,
            revision="b" * 40,
            baseline_sha="a" * 64,
            candidate_sha="c" * 64,
        )
        self.assertFalse(value["qualified"])

    def test_aggregate_binds_gpu_and_artifact_identity(self):
        reports = [
            report(case, gpu)
            for case, gpu in zip(RUNNER.CASES, (1, 2, 3))
        ]
        reports[0]["visible_physical_gpu"] = 3
        reports[1]["candidate_extension"]["sha256"] = "d" * 64
        value = RUNNER.aggregate(
            reports[:-1],
            revision="b" * 40,
            baseline_sha="a" * 64,
            candidate_sha="c" * 64,
        )
        self.assertFalse(value["qualified"])
        self.assertTrue(value["reasons"])


if __name__ == "__main__":
    unittest.main()
