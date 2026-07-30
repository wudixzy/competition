from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
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


if __name__ == "__main__":
    unittest.main()
