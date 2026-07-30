from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_m1_140_activation_capture.py"
SPEC = importlib.util.spec_from_file_location("m1_140_capture", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class M1140CaptureLifecycleTest(unittest.TestCase):

    def _runner(self, run_root: Path):
        runner = MODULE.CaptureRunner(SimpleNamespace(
            run_root=run_root,
            instance="unit-instance",
            profile="qualification",
            targets="32768,65536,131072",
            contexts="24576,57344,122880",
            ordinals="0,4,9",
        ))
        runner.run_id = "unit-run"
        runner.source_revision = "a" * 40
        runner.validate = mock.Mock()
        runner.prepare = mock.Mock()
        runner.verify_runtime = mock.Mock()
        runner.run_capture_requests = mock.Mock()
        runner.qualify_bank = mock.Mock()
        runner.cleanup_service = mock.Mock()
        runner.compare_preflights = mock.Mock()
        runner.source_unchanged = mock.Mock()
        runner.write_status = mock.Mock()
        return runner

    def test_synchronous_command_is_reaped_normally(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            returncode = MODULE._run_to_files(
                [sys.executable, "-c", "print('ok')"],
                root / "stdout",
                root / "stderr",
                timeout_s=5,
            )
            self.assertEqual(returncode, 0)
            self.assertEqual(
                (root / "stdout").read_text(encoding="ascii"),
                "ok\n",
            )

    def test_scoped_cleanup_sends_term_before_kill(self):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import signal,sys,time;"
                    "signal.signal(signal.SIGTERM,lambda *_: sys.exit(0));"
                    "time.sleep(30)"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        starttime = MODULE._read_starttime(process.pid)
        observed_signals = []
        real_killpg = MODULE.os.killpg

        def record_killpg(pgid, signum):
            observed_signals.append(signum)
            return real_killpg(pgid, signum)

        try:
            with mock.patch.object(
                    MODULE.os, "killpg", side_effect=record_killpg):
                self.assertTrue(
                    MODULE._stop_process_group(process, starttime))
            self.assertIsNotNone(process.returncode)
            self.assertEqual(observed_signals, [signal.SIGTERM])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

    def test_lifecycle_constants_and_process_isolation_are_frozen(self):
        source = SCRIPT.read_text(encoding="ascii")
        self.assertEqual(MODULE.TERM_GRACE_S, 60.0)
        self.assertEqual(MODULE.KILL_GRACE_S, 20.0)
        self.assertIn("start_new_session=True", source)
        self.assertIn("os.killpg", source)
        self.assertNotIn("pkill", source)

    def test_fatal_scan_failure_does_not_skip_final_gpu_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            runner.start_service = mock.Mock(
                side_effect=RuntimeError("startup interrupted"))
            runner.run_postflight = mock.Mock()
            runner.run_preflight = mock.Mock()
            runner.scan_fatal = mock.Mock(
                return_value={"worker_loss": 1})
            with mock.patch.object(MODULE, "append_event"):
                returncode = runner.run()

            self.assertEqual(returncode, 1)
            self.assertEqual(
                runner.run_postflight.call_args_list,
                [mock.call("postflight_before"),
                 mock.call("postflight_after")],
            )
            self.assertEqual(
                runner.run_preflight.call_args_list,
                [mock.call("preflight_before"),
                 mock.call("preflight_after")],
            )
            runner.compare_preflights.assert_called_once_with()
            runner.scan_fatal.assert_called_once_with()
            runner.source_unchanged.assert_called_once_with()
            self.assertEqual(runner.failed_stage, "service_startup")
            self.assertEqual(runner.current_stage, "service_startup")
            runner.write_status.assert_called_once_with(1)

    def test_postflight_failure_still_runs_fatal_and_source_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = self._runner(Path(temporary))
            runner.start_service = mock.Mock()

            def postflight(label):
                if label == "postflight_after":
                    raise RuntimeError("residual worker")

            runner.run_postflight = mock.Mock(side_effect=postflight)
            runner.run_preflight = mock.Mock()
            runner.scan_fatal = mock.Mock(return_value={"worker_loss": 0})
            with (
                mock.patch.object(MODULE, "append_event"),
                mock.patch.object(MODULE, "_health", return_value=True),
            ):
                returncode = runner.run()

            self.assertEqual(returncode, 1)
            self.assertEqual(
                runner.run_preflight.call_args_list,
                [mock.call("preflight_before")],
            )
            runner.compare_preflights.assert_not_called()
            runner.scan_fatal.assert_called_once_with()
            runner.source_unchanged.assert_called_once_with()
            self.assertIsNone(runner.gates["preflight_after"])
            self.assertIsNone(runner.gates["preflight_comparison"])
            self.assertEqual(runner.failed_stage, "postflight_after")


if __name__ == "__main__":
    unittest.main()
