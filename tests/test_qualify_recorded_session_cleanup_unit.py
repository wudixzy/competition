from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "qualify_recorded_session_cleanup.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_recorded_session_cleanup", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def action(identity: Path, pid: int) -> dict:
    return {
        "identity": str(identity.resolve()),
        "pid": pid,
        "pgid": pid,
        "sid": pid,
        "term_sent": False,
        "kill_sent": False,
        "initial_live_count": 0,
        "initial_escaped_count": 0,
        "token_scan_error_count": 0,
        "final_live_count": 0,
        "outcome": "already_quiescent",
    }


def recovery(identities: list[Path]) -> dict:
    return {
        "schema": "bi100-recorded-session-cleanup-v1",
        "version": 1,
        "qualified": True,
        "reasons": [],
        "identity_count": len(identities),
        "actions": [
            action(identity, 1000 + index)
            for index, identity in enumerate(identities)
        ],
        "term_grace_s": 60.0,
        "kill_grace_s": 20.0,
        "complete_token_scan_required": True,
        "privacy": {
            "command_lines_recorded": False,
            "environment_recorded": False,
        },
    }


class RecordedSessionCleanupQualificationUnitTest(unittest.TestCase):

    def setUp(self) -> None:
        self.identities = [
            Path("/tmp/control_child_identity.json"),
            Path("/tmp/candidate_child_identity.json"),
        ]

    def test_clean_quiescent_sessions_qualify(self) -> None:
        report = MODULE.qualify(
            recovery(self.identities), self.identities)
        self.assertTrue(report["qualified"], report["reasons"])
        self.assertFalse(report["emergency_recovery_used"])

    def test_emergency_term_invalidates_experiment(self) -> None:
        value = recovery(self.identities)
        value["actions"][0]["term_sent"] = True
        value["actions"][0]["initial_live_count"] = 1
        report = MODULE.qualify(value, self.identities)
        self.assertFalse(report["qualified"])
        self.assertTrue(report["emergency_recovery_used"])
        self.assertIn("action 1 required recovery", report["reasons"])

    def test_identity_order_and_count_are_exact(self) -> None:
        value = recovery(self.identities)
        value["actions"].reverse()
        report = MODULE.qualify(value, self.identities)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "recorded-session identity order differs", report["reasons"])

        value = recovery(self.identities[:1])
        report = MODULE.qualify(value, self.identities)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "recorded-session recovery contract differs", report["reasons"])

    def test_grace_period_or_scan_gap_fails_closed(self) -> None:
        value = recovery(self.identities)
        value["term_grace_s"] = 900.0
        value["complete_token_scan_required"] = False
        report = MODULE.qualify(value, self.identities)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "recorded-session recovery contract differs", report["reasons"])

    def test_token_scan_error_fails_closed(self) -> None:
        value = recovery(self.identities)
        value["actions"][1]["token_scan_error_count"] = 1
        report = MODULE.qualify(value, self.identities)
        self.assertFalse(report["qualified"])
        self.assertIn("action 2 required recovery", report["reasons"])

    def test_malformed_identity_path_fails_without_raising(self) -> None:
        value = recovery(self.identities)
        value["actions"][0]["identity"] = "bad\0path"
        report = MODULE.qualify(value, self.identities)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "action 1 identity path is malformed", report["reasons"])

    def test_cli_atomically_binds_the_parsed_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities = [
                root / "control_child_identity.json",
                root / "candidate_child_identity.json",
            ]
            source = root / "recovery.json"
            output = root / "qualified.json"
            payload = (
                json.dumps(recovery(identities), sort_keys=True) + "\n"
            ).encode("ascii")
            source.write_bytes(payload)
            with redirect_stdout(io.StringIO()):
                rc = MODULE.main([
                    str(source),
                    "--expected-identity", str(identities[0]),
                    "--expected-identity", str(identities[1]),
                    "--out", str(output),
                ])
            self.assertEqual(rc, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["qualified"])
            self.assertEqual(
                report["input_sha256"],
                hashlib.sha256(payload).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
