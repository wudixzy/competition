from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_m1_142_l1_subset_screen.py"
SPEC = importlib.util.spec_from_file_location("m1_142_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def cell(case: str, gpu: int, sha: str, speedup: float = 2.0):
    return {
        "schema": MODULE.CELL_SCHEMA,
        "case": case,
        "visible_physical_gpu": gpu,
        "extension": {"sha256": sha},
        "evaluation": {"qualified": True, "reasons": []},
        "timings": {"speedup": speedup},
        "numerical": {
            "finite": True,
            "output_relative_l2": 1e-6,
            "lse_relative_l2": 1e-7,
            "output_max_abs": 1e-5,
        },
    }


class M1142L1SubsetScreenTest(unittest.TestCase):

    def test_gpu_parser_rejects_invalid_or_excess_indices(self):
        self.assertEqual(MODULE.parse_gpus("2,3"), [2, 3])
        for value in ("", "2,2", "-1", "0,1,2,3,4", "x"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    MODULE.parse_gpus(value)

    def test_two_gpu_waves_can_pass_screen_but_not_full_l1(self):
        sha = "a" * 64
        reports = [
            cell(case, [2, 3][index % 2], sha)
            for index, case in enumerate(MODULE.CASES)
        ]
        result = MODULE.aggregate_cell_reports(
            reports=reports,
            gpus=[2, 3],
            extension_sha=sha,
        )
        self.assertTrue(result["screen_qualified"], result["reasons"])
        self.assertFalse(result["full_l1_contract_satisfied"])
        self.assertTrue(
            result["authorization"]["four_gpu_l1_rerun_authorized"])
        self.assertFalse(
            result["authorization"]["l2_capture_authorized"])

    def test_four_gpu_single_wave_can_authorize_only_l2(self):
        sha = "b" * 64
        reports = [
            cell(case, index, sha)
            for index, case in enumerate(MODULE.CASES)
        ]
        result = MODULE.aggregate_cell_reports(
            reports=reports,
            gpus=[0, 1, 2, 3],
            extension_sha=sha,
        )
        self.assertTrue(result["screen_qualified"], result["reasons"])
        self.assertTrue(result["full_l1_contract_satisfied"])
        self.assertTrue(result["authorization"]["l2_capture_authorized"])
        self.assertFalse(
            result["authorization"]["main_or_yaml_change_authorized"])

    def test_missing_bad_or_slow_cells_fail_closed(self):
        sha = "c" * 64
        reports = [
            cell(case, index, sha)
            for index, case in enumerate(MODULE.CASES[:-1])
        ]
        reports[0]["timings"]["speedup"] = 1.49
        result = MODULE.aggregate_cell_reports(
            reports=reports,
            gpus=[0, 1, 2, 3],
            extension_sha=sha,
        )
        self.assertFalse(result["screen_qualified"])
        self.assertFalse(result["authorization"]["l2_capture_authorized"])

    def test_aggregate_independently_rejects_bad_numerics(self):
        sha = "d" * 64
        reports = [
            cell(case, [2, 3][index % 2], sha)
            for index, case in enumerate(MODULE.CASES)
        ]
        reports[0]["numerical"]["output_relative_l2"] = 1.01e-5
        reports[1]["numerical"]["lse_relative_l2"] = float("nan")
        reports[2]["numerical"]["output_max_abs"] = 1.01e-3
        result = MODULE.aggregate_cell_reports(
            reports=reports,
            gpus=[2, 3],
            extension_sha=sha,
        )
        self.assertFalse(result["screen_qualified"])
        self.assertEqual(len(result["reasons"]), 3)

    def test_case_to_gpu_assignment_is_frozen(self):
        sha = "e" * 64
        reports = [
            cell(case, [3, 2][index % 2], sha)
            for index, case in enumerate(MODULE.CASES)
        ]
        result = MODULE.aggregate_cell_reports(
            reports=reports,
            gpus=[2, 3],
            extension_sha=sha,
        )
        self.assertFalse(result["screen_qualified"])

    def test_nonfinite_values_are_not_emitted_as_nonstandard_json(self):
        sha = "f" * 64
        reports = [
            cell(case, [2, 3][index % 2], sha)
            for index, case in enumerate(MODULE.CASES)
        ]
        reports[0]["numerical"]["output_relative_l2"] = float("nan")
        result = MODULE.aggregate_cell_reports(
            reports=reports,
            gpus=[2, 3],
            extension_sha=sha,
        )
        self.assertIsNone(result["rows"][0]["output_relative_l2"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            MODULE._atomic_json(path, result)
            decoded = json.loads(
                path.read_text(encoding="ascii"),
                parse_constant=lambda value: self.fail(
                    f"nonstandard JSON constant {value}"),
            )
        self.assertIsNone(decoded["rows"][0]["output_relative_l2"])

    def test_synchronous_child_is_reaped_from_managed_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            returncode = MODULE._run_to_files(
                [sys.executable, "-c", "print('ok')"],
                root / "stdout",
                root / "stderr",
                label="unit_child",
                timeout_s=5,
                environment=MODULE._base_environment(),
            )
            self.assertEqual(returncode, 0)
            self.assertEqual(MODULE._ACTIVE_CHILDREN, [])

    def test_lifecycle_contract_is_scoped_and_term_first(self):
        source = SCRIPT.read_text(encoding="ascii")
        self.assertEqual(MODULE.TERM_GRACE_S, 60.0)
        self.assertEqual(MODULE.KILL_GRACE_S, 20.0)
        self.assertIn("start_new_session=True", source)
        self.assertIn("os.killpg", source)
        self.assertIn("signal.SIGTERM", source)
        self.assertIn("signal.SIGKILL", source)
        self.assertIn("--kill-after=90s", source)
        self.assertNotIn("pkill", source)


if __name__ == "__main__":
    unittest.main()
