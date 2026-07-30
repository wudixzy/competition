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
    "bench_m1_149_ttft_p90_prefill",
    TESTS / "bench_m1_149_ttft_p90_prefill.py",
)
RUNNER = _load(
    "run_m1_149_ttft_p90_prefill_grid",
    SCRIPTS / "run_m1_149_ttft_p90_prefill_grid.py",
)


def report(case: str, gpu: int, *, speedup: float = 1.5) -> dict:
    context_len, query_len = CELL.CASES[case]
    value = {
        "schema": CELL.SCHEMA,
        "version": 1,
        "source_commit": "b" * 40,
        "case": case,
        "context_len": context_len,
        "query_len": query_len,
        "total_kv_len": context_len + query_len,
        "seed": CELL.production.SEED,
        "warmups": CELL.production.WARMUPS,
        "trials": CELL.production.TRIALS,
        "visible_physical_gpu": gpu,
        "extension": {"sha256": "a" * 64},
        "timings": {
            "reference": {"cuda_median_ms": 3.0},
            "candidate": {"cuda_median_ms": 2.0},
            "speedup": speedup,
        },
        "numerical": {
            "finite": True,
            "output_relative_l2": 5.0e-6,
            "lse_relative_l2": 2.0e-8,
            "output_max_abs": 2.0e-4,
        },
        "authorization": {
            "short_tp4_p90_screen_authorized": False,
            "l2_capture_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }
    value["evaluation"] = CELL.evaluate(value)
    return value


class M1149TtftP90PrefillTest(unittest.TestCase):

    def test_grid_matches_chunked_prefill_positions(self) -> None:
        self.assertEqual(len(CELL.CASES), 8)
        totals = [
            context + query
            for context, query in CELL.CASES.values()
        ]
        self.assertEqual(
            totals,
            [8176, 16368, 24560, 32752, 40944, 49136, 57328, 65520],
        )

    def test_cell_contract_qualifies_and_fails_closed(self) -> None:
        case = next(iter(CELL.CASES))
        value = report(case, 1)
        self.assertTrue(value["evaluation"]["qualified"])
        value["timings"]["speedup"] = 1.1
        self.assertFalse(CELL.evaluate(value)["qualified"])
        value["timings"]["speedup"] = 1.5
        value["numerical"]["output_relative_l2"] = 2.0e-5
        self.assertFalse(CELL.evaluate(value)["qualified"])

    def test_aggregate_binds_cases_gpus_and_extension(self) -> None:
        gpus = [1, 2, 3]
        reports = [
            report(case, gpus[index % len(gpus)])
            for index, case in enumerate(RUNNER.CASES)
        ]
        value = RUNNER.aggregate(
            reports,
            gpus=gpus,
            extension_sha="a" * 64,
            source_revision="b" * 40,
        )
        self.assertTrue(value["qualified"], value)
        self.assertTrue(
            value["authorization"][
                "short_tp4_p90_screen_authorized"])
        self.assertFalse(
            value["authorization"]["l2_capture_authorized"])

    def test_aggregate_rejects_missing_or_misassigned_case(self) -> None:
        gpus = [1, 2, 3]
        reports = [
            report(case, gpus[index % len(gpus)])
            for index, case in enumerate(RUNNER.CASES)
        ]
        reports[0]["visible_physical_gpu"] = 3
        value = RUNNER.aggregate(
            reports[:-1],
            gpus=gpus,
            extension_sha="a" * 64,
            source_revision="b" * 40,
        )
        self.assertFalse(value["qualified"])
        self.assertTrue(value["reasons"])


if __name__ == "__main__":
    unittest.main()
