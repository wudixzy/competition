from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/qualify_m1_137_outer_lifecycle.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_m1_137_outer_lifecycle_unit", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def aggregate() -> dict:
    return {
        "schema": M.AGGREGATE_SCHEMA,
        "version": 1,
        "qualified": True,
        "ifeval_two_point_capability_surface_statistically_qualified": True,
        "ifeval_two_point_capability_surface_authorized": False,
        "outer_lifecycle_pending": True,
        "performance_authorized": False,
        "default_change_authorized": False,
        "yaml_change_authorized": False,
        "main_merge_authorized": False,
        "production_promotion_authorized": False,
        "reasons": [],
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_sample_outcomes": False,
            "contains_credentials": False,
        },
    }


def prepare(root: Path) -> None:
    for name in M.EXPECTED_RCS:
        (root / f"{name}.rc").write_text("0\n", encoding="utf-8")
    write_json(root / "aggregate.json", aggregate())
    recovery = {
        "schema": "bi100-recorded-session-cleanup-v1",
        "version": 1,
        "qualified": True,
        "reasons": [],
    }
    write_json(root / "orchestrator_recovery.json", recovery)
    write_json(root / "orchestrator_recovery_clean.json", {
        "schema": "bi100-recorded-session-cleanup-qualification-v1",
        "version": 1,
        "qualified": True,
        "reasons": [],
        "emergency_recovery_used": False,
        "production_promotion_authorized": False,
        "input_sha256": M.sha256(root / "orchestrator_recovery.json"),
    })
    write_json(root / "orchestrator_postflight.json", {
        "schema": "bi100-service-postflight-v1",
        "version": 1,
        "qualified": True,
        "gpu_indices": [0, 1, 2, 3],
        "missing_devices": [],
        "api_server_pids": [],
        "worker_pids": [],
        "gpu_processes": [],
        "scan_errors": [],
        "settling": {
            "timeout_s": 30.0,
            "sample_interval_s": 1.0,
            "required_clean_samples": 3,
            "final_clean_streak": 3,
            "attempts": 3,
        },
        "privacy": {
            "command_lines_recorded": False,
            "environment_recorded": False,
        },
    })
    write_json(root / "orchestrator_preflight_after.json", {
        "schema": "bi100-gpu-preflight-v1",
        "version": 1,
        "gpus": [0, 1, 2, 3],
        "matmul_size": 1024,
        "timeout_s": 25.0,
        "ok": True,
        "results": [
            {
                "gpu": gpu,
                "ok": True,
                "stage": "done",
                "returncode": 0,
            }
            for gpu in range(4)
        ],
    })
    (root / "orchestrator_fatal_scan.txt").write_bytes(b"")
    (root / "orchestrator_timeout_scan.txt").write_bytes(b"")


class QualifyM1137OuterLifecycleTest(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        prepare(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def qualify(self) -> dict:
        return M.qualify(
            self.root,
            self.root / "aggregate.json",
            aggregate_recomputer=lambda _root: aggregate(),
        )

    def test_clean_lifecycle_authorizes_only_capability_surface(self) -> None:
        value = self.qualify()
        self.assertTrue(value["qualified"], value)
        self.assertTrue(
            value["ifeval_two_point_capability_surface_authorized"])
        for name in (
            "performance_authorized",
            "default_change_authorized",
            "yaml_change_authorized",
            "main_merge_authorized",
            "production_promotion_authorized",
        ):
            self.assertFalse(value[name])
        self.assertTrue(all(
            isinstance(digest, str) and len(digest) == 64
            for digest in value["evidence"].values()
        ))

    def test_outer_lifecycle_failures_reject(self) -> None:
        for rc_name in (
            "orchestrator_postflight",
            "orchestrator_fatal_scan",
            "orchestrator_timeout_scan",
        ):
            with self.subTest(rc_name=rc_name):
                prepare(self.root)
                (self.root / f"{rc_name}.rc").write_text(
                    "1\n", encoding="utf-8")
                value = self.qualify()
                self.assertFalse(value["qualified"], value)
                self.assertFalse(
                    value[
                        "ifeval_two_point_capability_surface_authorized"])

    def test_nonempty_fatal_and_timeout_scans_reject(self) -> None:
        for name in (
            "orchestrator_fatal_scan.txt",
            "orchestrator_timeout_scan.txt",
        ):
            with self.subTest(name=name):
                prepare(self.root)
                (self.root / name).write_text(
                    "failure marker\n", encoding="utf-8")
                value = self.qualify()
                self.assertFalse(value["qualified"], value)
                self.assertIn(f"{name} is not empty", value["reasons"])

    def test_precleanup_aggregate_cannot_self_authorize(self) -> None:
        changed = aggregate()
        changed["ifeval_two_point_capability_surface_authorized"] = True
        write_json(self.root / "aggregate.json", changed)
        value = self.qualify()
        self.assertFalse(value["qualified"], value)
        self.assertFalse(
            value["ifeval_two_point_capability_surface_authorized"])

    def test_forged_aggregate_cannot_replace_recomputed_arm_evidence(
        self,
    ) -> None:
        forged = aggregate()
        forged["strict_zero_stratum_qualified"] = True
        write_json(self.root / "aggregate.json", forged)
        value = self.qualify()
        self.assertFalse(value["qualified"], value)
        self.assertIn(
            "M1-137 aggregate differs from recomputed arm evidence",
            value["reasons"],
        )

    def test_default_recompute_reads_retained_arm_evidence(self) -> None:
        expected = aggregate()
        with mock.patch.object(
            M.m1_137,
            "compare_from_paths",
            return_value=expected,
        ) as recompute:
            self.assertEqual(M._recompute_aggregate(self.root), expected)
        recompute.assert_called_once_with(
            control_root=self.root / "control",
            candidate_root=self.root / "candidate",
            score_comparison=(
                self.root / "ifeval_score_comparison.json"),
            exact_comparison=(
                self.root / "ifeval_exact_comparison.json"),
            paired_noninferiority=(
                self.root / "ifeval_paired_noninferiority.json"),
        )

    def test_recovery_binding_tamper_rejects(self) -> None:
        changed = json.loads(
            (self.root / "orchestrator_recovery.json").read_text(
                encoding="utf-8"))
        changed["extra"] = True
        write_json(self.root / "orchestrator_recovery.json", changed)
        value = self.qualify()
        self.assertFalse(value["qualified"], value)
        self.assertIn(
            "recorded-session cleanup did not qualify", value["reasons"])

    def test_final_rescan_rejects_fatal_log_and_timeout_rc(self) -> None:
        for relative, payload, expected in (
            (
                "control/server.log",
                "CUDA error during unit probe\n",
                "final fatal rescan found 1 affected files",
            ),
            (
                "control/measurement.rc",
                "124\n",
                "final timeout rescan found 1 timeout return codes",
            ),
        ):
            with self.subTest(relative=relative):
                prepare(self.root)
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload, encoding="utf-8")
                value = self.qualify()
                self.assertFalse(value["qualified"], value)
                self.assertIn(expected, value["reasons"])
                path.unlink()


class M1137OuterRunnerStaticTest(unittest.TestCase):

    def test_final_qualification_runs_after_outer_scans(self) -> None:
        runner = (
            ROOT / "scripts/run_m1_85_admission64_quality_ab.sh"
        ).read_text(encoding="utf-8")
        finish = runner[runner.index("finish() {"):]
        final_index = finish.index(
            "qualify_m1_137_outer_lifecycle.py")
        self.assertLess(
            finish.index("scan_orchestrator_fatal_logs"), final_index)
        self.assertLess(
            finish.index("scan_orchestrator_timeouts"), final_index)
        self.assertIn('"final_qualification"', runner)
        self.assertIn("final_qualification_sha256", runner)
        qualifier = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("compare_from_paths", qualifier)
        self.assertIn("_rescan_current_artifacts", qualifier)


if __name__ == "__main__":
    unittest.main()
