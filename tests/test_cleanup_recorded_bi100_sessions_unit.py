#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cleanup_recorded_bi100_sessions as cleanup  # noqa: E402


def process_starttime(pid: int) -> int:
    value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    return int(value[value.rfind(")") + 2:].split()[19])


class CleanupRecordedBi100SessionsUnitTest(unittest.TestCase):
    def _spawn(
        self,
        *,
        token: str,
        ignore_term: bool = False,
    ) -> subprocess.Popen[bytes]:
        command = "trap '' TERM; exec sleep 60" if ignore_term else "exec sleep 60"
        environment = os.environ.copy()
        environment["BI100_PROCESS_SESSION_TOKEN"] = token
        process = subprocess.Popen(
            ["bash", "-c", command],
            env=environment,
            start_new_session=True,
        )
        for _ in range(100):
            if Path(f"/proc/{process.pid}/environ").exists():
                break
            time.sleep(0.01)
        return process

    @staticmethod
    def _identity(path: Path, process: subprocess.Popen[bytes], token: str) -> None:
        path.write_text(json.dumps({
            "schema": cleanup.IDENTITY_SCHEMA,
            "version": 1,
            "pid": process.pid,
            "pgid": process.pid,
            "sid": process.pid,
            "starttime_ticks": process_starttime(process.pid),
            "session_token": token,
        }) + "\n", encoding="utf-8")

    def test_recorded_group_gets_term_and_quiesces(self) -> None:
        token = "a" * 32
        process = self._spawn(token=token)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                identity = Path(temporary) / "identity.json"
                self._identity(identity, process, token)
                report = cleanup.recover(
                    [identity], term_grace_s=2.0, kill_grace_s=1.0,
                    require_complete_token_scan=False)
            self.assertTrue(report["qualified"], report["reasons"])
            self.assertTrue(report["actions"][0]["term_sent"])
            self.assertFalse(report["actions"][0]["kill_sent"])
            process.wait(timeout=3)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)

    def test_term_ignoring_group_uses_kill_only_after_grace(self) -> None:
        token = "b" * 32
        process = self._spawn(token=token, ignore_term=True)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                identity = Path(temporary) / "identity.json"
                self._identity(identity, process, token)
                started = time.monotonic()
                report = cleanup.recover(
                    [identity], term_grace_s=0.2, kill_grace_s=1.0,
                    require_complete_token_scan=False)
                elapsed = time.monotonic() - started
            self.assertTrue(report["qualified"], report["reasons"])
            self.assertGreaterEqual(elapsed, 0.18)
            self.assertTrue(report["actions"][0]["term_sent"])
            self.assertTrue(report["actions"][0]["kill_sent"])
            process.wait(timeout=3)
            self.assertEqual(process.returncode, -9)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)

    def test_token_mismatch_refuses_to_signal_group(self) -> None:
        process = self._spawn(token="c" * 32)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                identity = Path(temporary) / "identity.json"
                self._identity(identity, process, "d" * 32)
                report = cleanup.recover(
                    [identity], term_grace_s=0.0, kill_grace_s=0.0,
                    require_complete_token_scan=False)
            self.assertFalse(report["qualified"])
            self.assertIn(
                "process session token differs", report["reasons"][0])
            self.assertIsNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)

    def test_token_bearing_process_that_escaped_group_is_cleaned(self) -> None:
        token = "e" * 32
        environment = os.environ.copy()
        environment["BI100_PROCESS_SESSION_TOKEN"] = token
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "escaped.pid"
            process = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    f"setsid sleep 60 & echo $! > {pid_path!s}; exec sleep 60",
                ],
                env=environment,
                start_new_session=True,
            )
            escaped_pid = None
            try:
                for _ in range(100):
                    if pid_path.is_file():
                        escaped_pid = int(
                            pid_path.read_text(encoding="ascii").strip())
                        break
                    time.sleep(0.01)
                self.assertIsNotNone(escaped_pid)
                identity = Path(temporary) / "identity.json"
                self._identity(identity, process, token)
                report = cleanup.recover(
                    [identity], term_grace_s=2.0, kill_grace_s=1.0,
                    require_complete_token_scan=False)
                self.assertTrue(report["qualified"], report["reasons"])
                self.assertGreaterEqual(
                    report["actions"][0]["initial_escaped_count"], 1)
                self.assertEqual(report["actions"][0]["final_live_count"], 0)
                process.wait(timeout=3)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=3)
                if escaped_pid is not None:
                    try:
                        os.kill(escaped_pid, 9)
                    except ProcessLookupError:
                        pass

    def test_incomplete_token_scan_fails_closed(self) -> None:
        token = "f" * 32
        process = self._spawn(token=token)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                identity = Path(temporary) / "identity.json"
                self._identity(identity, process, token)
                with mock.patch.object(
                        cleanup, "_token_members", return_value=([], 1)):
                    report = cleanup.recover(
                        [identity],
                        term_grace_s=0.0,
                        kill_grace_s=0.0,
                        require_complete_token_scan=True,
                    )
            self.assertFalse(report["qualified"])
            self.assertIn(
                "token process scan was incomplete", report["reasons"][0])
            self.assertIsNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
