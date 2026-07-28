from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
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
