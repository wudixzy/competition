from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import compare_bi100_preflights


SCRIPT = ROOT / "tests" / "bi100_parallel_preflight.py"
SPEC = importlib.util.spec_from_file_location(
    "bi100_parallel_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ParallelBi100PreflightTest(unittest.TestCase):

    def test_parse_gpus_rejects_empty_and_duplicates(self):
        self.assertEqual(MODULE.parse_gpus("0, 2,3"), [0, 2, 3])
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.parse_gpus("")
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.parse_gpus("0,0")

    def test_parallel_runner_collects_independent_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake_preflight.py"
            fake.write_text(textwrap.dedent("""
                import argparse
                import json
                from pathlib import Path

                parser = argparse.ArgumentParser()
                parser.add_argument("--gpus", type=int)
                parser.add_argument("--timeout-s")
                parser.add_argument("--matmul-size")
                parser.add_argument("--json-out", type=Path)
                args = parser.parse_args()
                value = {
                    "results": [{
                        "gpu": args.gpus,
                        "ok": True,
                        "stage": "done",
                        "returncode": 0,
                        "device_name": "BI100",
                        "device_capability": [10, 0],
                        "free": 1024,
                        "total": 2048,
                        "checksum": 512,
                    }]
                }
                args.json_out.write_text(json.dumps(value))
            """).strip() + "\n", encoding="ascii")
            summary = MODULE.run_parallel(
                gpus=[0, 2],
                timeout_s=1,
                matmul_size=8,
                serial_script=fake,
                work_dir=root / "work",
            )

        self.assertTrue(summary["ok"])
        self.assertTrue(summary["parallel"])
        self.assertEqual(summary["schema"], "bi100-gpu-preflight-v1")
        self.assertEqual(summary["runner"], "parallel-single-gpu")
        comparison = compare_bi100_preflights.compare(
            [("before", summary), ("after", summary)],
            expected_gpus=(0, 2),
        )
        self.assertTrue(comparison["qualified"], comparison["reasons"])
        self.assertEqual(
            [result["gpu"] for result in summary["results"]],
            [0, 2],
        )
        self.assertTrue(summary["cleanup_reaped"])

    def test_cleanup_contract_is_term_first_with_full_grace(self):
        source = SCRIPT.read_text(encoding="ascii")
        self.assertGreaterEqual(MODULE.TERM_GRACE_S, 45.0)
        self.assertIn("signal.SIGTERM", source)
        self.assertIn("signal.SIGKILL", source)
        self.assertIn("start_new_session=True", source)

    def test_funnel_runners_do_not_cross_python_abis(self):
        for relative in (
            "scripts/run_m1_140_activation_capture.py",
            "scripts/run_m1_141_short_tp4_screen.py",
        ):
            source = (ROOT / relative).read_text(encoding="ascii")
            self.assertNotIn('"/usr/bin/python3"', source)
            self.assertIn("sys.executable", source)

    def test_outer_timeouts_allow_inner_cleanup_to_finish(self):
        capture = (
            ROOT / "scripts" / "run_m1_140_activation_capture.py"
        ).read_text(encoding="ascii")
        replay = (
            ROOT / "scripts" / "run_m1_140_activation_replay.sh"
        ).read_text(encoding="ascii")
        self.assertIn('"--kill-after=90s"', capture)
        self.assertNotIn("--kill-after=70s", capture)
        self.assertNotIn("--kill-after=70s", replay)
        self.assertGreaterEqual(replay.count("--kill-after=90s"), 3)


if __name__ == "__main__":
    unittest.main()
