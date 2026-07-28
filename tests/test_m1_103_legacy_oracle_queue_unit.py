from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests import qualify_m1_103_legacy_oracle_queue as module


REVISION = "1" * 40
TOKEN_A = "a" * 32
TOKEN_B = "b" * 32
RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_m1_103_legacy_oracle_queue.sh"
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prefix_report(qualified: bool) -> dict:
    return {
        "schema": module.PREFIX_SCHEMA,
        "version": 1,
        "frozen_artifacts": module.PREFIX_FROZEN,
        "config": {
            "production_query_len": 8176,
            "primary_context": 65536,
            "partial_context": 65552,
            "minimum_primary_reduction": 0.15,
        },
        "summary": {
            "qualified": qualified,
            "reasons": [] if qualified else ["valid negative"],
            "decision": {
                "next_token_gate_authorized": qualified,
                "service_integration_authorized": False,
                "production_promotion_authorized": False,
                "yaml_change_authorized": False,
                "main_merge_authorized": False,
            },
        },
    }


def wmma_report(qualified: bool) -> dict:
    return {
        "schema": module.WMMA_SCHEMA,
        "version": 1,
        "frozen_artifacts": module.WMMA_FROZEN,
        "extension_sha256": "c" * 64,
        "config": {
            "tiles": 128,
            "head_dim": 256,
            "minimum_qk_speedup": 1.5,
        },
        "summary": {
            "qualified": qualified,
            "reasons": [] if qualified else ["valid negative"],
            "decision": {
                "integration_benefit_gate_authorized": qualified,
                "service_integration_authorized": False,
                "production_promotion_authorized": False,
                "yaml_change_authorized": False,
                "main_merge_authorized": False,
            },
        },
    }


def identity(pid: int, token: str) -> dict:
    return {
        "schema": "bi100-process-session-v1",
        "version": 1,
        "pid": pid,
        "pgid": pid,
        "sid": pid,
        "starttime_ticks": pid * 10,
        "session_token": token,
    }


def populate(
    root: Path,
    *,
    prefix_qualified: bool = False,
    wmma_qualified: bool = True,
) -> None:
    for name in module.ZERO_GATES:
        (root / f"{name}.rc").write_text("0\n", encoding="ascii")
    (root / "prefix.rc").write_text(
        f"{0 if prefix_qualified else 1}\n", encoding="ascii")
    (root / "wmma.rc").write_text(
        f"{0 if wmma_qualified else 1}\n", encoding="ascii")
    (root / "source_revision.txt").write_text(
        REVISION + "\n", encoding="ascii")
    (root / "source_branch.txt").write_text(
        "exp/M1-103\n", encoding="ascii")
    (root / "instance.txt").write_text("private-bi100\n", encoding="ascii")
    (root / "prefix_gpu.txt").write_text("0\n", encoding="ascii")
    (root / "wmma_gpu.txt").write_text("1\n", encoding="ascii")
    (root / "stage.txt").write_text("completed\n", encoding="ascii")
    (root / "fatal_scan.txt").write_text("", encoding="ascii")
    (root / "timeout_scan.txt").write_text("", encoding="ascii")
    write_json(root / "prefix" / "report.json",
               prefix_report(prefix_qualified))
    write_json(root / "wmma" / "report.json",
               wmma_report(wmma_qualified))
    write_json(root / "prefix_identity.json", identity(101, TOKEN_A))
    write_json(root / "wmma_identity.json", identity(202, TOKEN_B))
    write_json(root / "recovery_clean.json", {
        "qualified": True,
        "emergency_recovery_used": False,
    })
    write_json(root / "postflight_before.json", {"qualified": True})
    write_json(root / "postflight.json", {"qualified": True})
    write_json(root / "preflight_comparison.json", {"qualified": True})


class M1103LegacyOracleQueueUnitTest(unittest.TestCase):

    def qualify(self, root: Path, runner_returncode: int = 0) -> dict:
        return module.qualify(
            root,
            expected_source_revision=REVISION,
            expected_prefix_gpu=0,
            expected_wmma_gpu=1,
            runner_returncode=runner_returncode,
        )

    def test_valid_negative_is_a_qualified_queue_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root, prefix_qualified=False, wmma_qualified=True)
            report = self.qualify(root)
            self.assertTrue(report["qualified"], report["reasons"])
            self.assertFalse(report["candidates"]["prefix"]["qualified"])
            self.assertTrue(report["candidates"]["wmma"]["qualified"])
            encoded = json.dumps(report, sort_keys=True)
            self.assertNotIn(TOKEN_A, encoded)
            self.assertNotIn(TOKEN_B, encoded)

    def test_runner_uses_parallel_scoped_lifecycle(self):
        source = RUNNER.read_text(encoding="utf-8")
        for marker in (
            "EXPERIMENT_TIMEOUT_S=7200",
            'start_child prefix "$PREFIX_GPU"',
            'start_child wmma "$WMMA_GPU"',
            "wait_for_children",
            '"${CHILD_PGID[$label]}" "${CHILD_PID[$label]}" 60 20',
            'run_postflight "$RUN_ROOT/postflight_before"',
            'run_postflight "$RUN_ROOT/postflight"',
            'run_preflight "$RUN_ROOT/preflight_after"',
            "qualify_recorded_session_cleanup.py",
        ):
            self.assertIn(marker, source)
        for forbidden in ("pkill", "killall", "pkill -9", "kill -9"):
            self.assertNotIn(forbidden, source)

    def test_candidate_returncode_must_match_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root, prefix_qualified=False)
            (root / "prefix.rc").write_text("0\n", encoding="ascii")
            report = self.qualify(root)
            self.assertFalse(report["qualified"])
            self.assertIn(
                "prefix return code does not match report",
                report["reasons"],
            )

    def test_candidate_cannot_authorize_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root)
            value = json.loads(
                (root / "wmma" / "report.json").read_text())
            value["summary"]["decision"][
                "production_promotion_authorized"] = True
            write_json(root / "wmma" / "report.json", value)
            report = self.qualify(root)
            self.assertFalse(report["qualified"])
            self.assertTrue(any(
                "production_promotion_authorized" in reason
                for reason in report["reasons"]
            ))

    def test_lifecycle_or_scan_failure_rejects_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root)
            (root / "postflight.rc").write_text("1\n", encoding="ascii")
            (root / "fatal_scan.txt").write_text(
                "CUDA error\n", encoding="ascii")
            report = self.qualify(root)
            self.assertFalse(report["qualified"])
            self.assertIn(
                "one or more infrastructure/lifecycle gates failed",
                report["reasons"],
            )
            self.assertIn("fatal_scan.txt is not empty", report["reasons"])

    def test_identity_and_source_drift_reject_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            populate(root)
            value = identity(101, TOKEN_A)
            value["pgid"] = 999
            write_json(root / "prefix_identity.json", value)
            report = module.qualify(
                root,
                expected_source_revision="2" * 40,
                expected_prefix_gpu=0,
                expected_wmma_gpu=1,
                runner_returncode=0,
            )
            self.assertFalse(report["qualified"])
            self.assertIn("source revision differs", report["reasons"])
            self.assertIn(
                "prefix process identity differs",
                report["reasons"],
            )


if __name__ == "__main__":
    unittest.main()
