from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "exec_bi100_session.py"


def process_environment(pid: int) -> set[bytes]:
    return set(Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"))


class ExecBi100SessionUnitTest(unittest.TestCase):

    def test_identity_is_private_and_matches_new_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            identity = Path(temporary) / "identity.json"
            inherited = Path(temporary) / "inherited.txt"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(HELPER),
                    str(identity),
                    "--",
                    "/bin/sh",
                    "-c",
                    'printf "%s" "$BI100_PROCESS_SESSION_TOKEN" > "$1"',
                    "sh",
                    str(inherited),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            value = json.loads(identity.read_text(encoding="utf-8"))
            self.assertEqual(value["schema"], "bi100-process-session-v1")
            self.assertEqual(value["version"], 1)
            self.assertEqual(value["pid"], value["pgid"])
            self.assertEqual(value["pid"], value["sid"])
            self.assertIsInstance(value["starttime_ticks"], int)
            self.assertGreater(value["starttime_ticks"], 0)
            self.assertRegex(value["session_token"], r"^[0-9a-f]{32}$")
            self.assertEqual(
                inherited.read_text(encoding="ascii"),
                value["session_token"],
            )
            mode = stat.S_IMODE(identity.stat().st_mode)
            self.assertEqual(mode, 0o600)
            self.assertIn(
                "[BI100 SESSION] child subreaper active",
                completed.stdout,
            )

    def test_leader_proc_environment_contains_recorded_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            identity = Path(temporary) / "identity.json"
            child_pid_path = Path(temporary) / "child.pid"
            environment = os.environ.copy()
            environment["BI100_PROCESS_SESSION_TOKEN"] = "f" * 32
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(HELPER),
                    str(identity),
                    "--",
                    "/bin/sh",
                    "-c",
                    'printf "%s" "$$" > "$1"; exec sleep 60',
                    "sh",
                    str(child_pid_path),
                ],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while (
                    not identity.is_file()
                    or not child_pid_path.is_file()
                ):
                    if process.poll() is not None:
                        break
                    if time.monotonic() >= deadline:
                        self.fail("session identity did not become ready")
                    time.sleep(0.02)
                self.assertIsNone(process.poll())
                value = json.loads(identity.read_text(encoding="utf-8"))
                entries = process_environment(process.pid)
                expected = (
                    "BI100_PROCESS_SESSION_TOKEN="
                    f"{value['session_token']}"
                ).encode("ascii")
                self.assertIn(expected, entries)
                self.assertNotIn(
                    b"BI100_PROCESS_SESSION_TOKEN=" + b"f" * 32,
                    entries,
                )
                child_pid = int(
                    child_pid_path.read_text(encoding="ascii"))
                self.assertNotIn(
                    b"BI100_EXEC_SESSION_REEXEC_V1=1",
                    process_environment(child_pid),
                )
                self.assertIn(expected, process_environment(child_pid))
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.communicate(timeout=5)

    def test_nested_session_leaders_have_distinct_proc_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outer_identity = root / "outer.json"
            inner_identity = root / "inner.json"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(HELPER),
                    str(outer_identity),
                    "--",
                    sys.executable,
                    str(HELPER),
                    str(inner_identity),
                    "--",
                    "sleep",
                    "60",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while (
                    not outer_identity.is_file()
                    or not inner_identity.is_file()
                ):
                    if process.poll() is not None:
                        break
                    if time.monotonic() >= deadline:
                        self.fail("nested session identities did not become ready")
                    time.sleep(0.02)
                self.assertIsNone(process.poll())
                outer = json.loads(
                    outer_identity.read_text(encoding="utf-8"))
                inner = json.loads(
                    inner_identity.read_text(encoding="utf-8"))
                self.assertNotEqual(
                    outer["session_token"], inner["session_token"])
                for value in (outer, inner):
                    expected = (
                        "BI100_PROCESS_SESSION_TOKEN="
                        f"{value['session_token']}"
                    ).encode("ascii")
                    self.assertIn(
                        expected, process_environment(value["pid"]))

                os.killpg(inner["pgid"], signal.SIGTERM)
                _, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 143, stderr)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)

    def test_orphaned_grandchild_is_reaped_before_session_returns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = root / "identity.json"
            grandchild_path = root / "grandchild.pid"
            code = """
import os
from pathlib import Path
import sys
import time

pid = os.fork()
if pid == 0:
    time.sleep(0.2)
    os._exit(0)
Path(sys.argv[1]).write_text(str(pid), encoding="ascii")
os._exit(0)
"""
            completed = subprocess.run(
                [
                    sys.executable,
                    str(HELPER),
                    str(identity),
                    "--",
                    sys.executable,
                    "-c",
                    code,
                    str(grandchild_path),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            grandchild_pid = int(
                grandchild_path.read_text(encoding="ascii"))
            self.assertFalse(Path(f"/proc/{grandchild_pid}").exists())

    def test_group_term_is_forwarded_and_all_descendants_are_reaped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = root / "identity.json"
            grandchild_path = root / "grandchild.pid"
            code = """
import os
from pathlib import Path
import sys
import time

pid = os.fork()
if pid == 0:
    time.sleep(60)
    os._exit(0)
Path(sys.argv[1]).write_text(str(pid), encoding="ascii")
time.sleep(60)
"""
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(HELPER),
                    str(identity),
                    "--",
                    sys.executable,
                    "-c",
                    code,
                    str(grandchild_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while (
                    not identity.is_file()
                    or not grandchild_path.is_file()
                ):
                    if process.poll() is not None:
                        break
                    if time.monotonic() >= deadline:
                        self.fail("session descendants did not become ready")
                    time.sleep(0.02)
                self.assertIsNone(process.poll())
                value = json.loads(identity.read_text(encoding="utf-8"))
                self.assertEqual(value["pid"], process.pid)
                grandchild_pid = int(
                    grandchild_path.read_text(encoding="ascii"))

                os.killpg(process.pid, signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 143, stderr)
                self.assertIn(
                    "[BI100 SESSION] child subreaper active", stdout)
                self.assertFalse(
                    Path(f"/proc/{grandchild_pid}").exists())
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)

    def test_existing_identity_fails_before_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            identity = Path(temporary) / "identity.json"
            identity.write_text("{}\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(HELPER),
                    str(identity),
                    "--",
                    "/bin/sh",
                    "-c",
                    "exit 99",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("already exists", completed.stderr)


if __name__ == "__main__":
    unittest.main()
