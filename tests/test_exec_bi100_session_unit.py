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
