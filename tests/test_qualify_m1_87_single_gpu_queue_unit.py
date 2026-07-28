#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import qualify_m1_87_single_gpu_queue as qualifier  # noqa: E402


REVISION = "1" * 40
TREE = "2" * 64
MANIFEST = "3" * 64
MODEL = "/models/Qwen3.6-35B-A3B-diagnostic-4L-real"
RUNTIME = "/runtime/site-packages"


class M187QueueQualifierUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "m1_84").mkdir()
        (self.root / "m1_86" / "control").mkdir(parents=True)
        (self.root / "m1_86" / "candidate").mkdir(parents=True)
        self._write_qualified_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_qualified_fixture(self) -> None:
        authority = {
            "full_model_evaluated": False,
            "semantic_quality_evaluated": False,
            "performance_evaluated": False,
            "production_promotion_authorized": False,
        }
        overlay = {
            "schema": qualifier.OVERLAY_SCHEMA,
            "version": 1,
            "qualified": True,
            "source_revision": REVISION,
            "runtime_site_packages": RUNTIME,
            "runtime_tree_sha256": TREE,
        }
        self._write(
            self.root / "m1_89_runtime_overlay_identity.json", overlay)
        self._write(
            self.root / "m1_84" / "runtime_overlay_identity.json", overlay)
        self._write(
            self.root / "m1_86" / "runtime_overlay_identity.json", overlay)
        self._write(
            self.root / "m1_89_cache_namespace_runtime_gate.json",
            {
                "schema": qualifier.CACHE_NAMESPACE_SCHEMA,
                "version": 2,
                "qualified": True,
                "reasons": [],
                "source_revision": REVISION,
                "runtime_site_packages": RUNTIME,
                "block_manager_module_sha256": "4" * 64,
                "sequence_module_sha256": "5" * 64,
                "pillow_version": "11.3.0",
                "checks": {
                    name: True
                    for name in qualifier.CACHE_NAMESPACE_CHECKS
                },
                "error_types": {},
                "privacy": {
                    "contains_image_bytes": False,
                    "contains_namespace_digest": False,
                    "contains_request_id": False,
                    "contains_prompt_or_output": False,
                    "contains_credentials": False,
                },
                "gpu_execution_required": False,
                "model_execution_performed": False,
                "production_promotion_authorized": False,
            },
        )
        runtime_identity = {
            "source_revision": REVISION,
            "physical_gpus": ["3"],
            "tensor_parallel_size": 1,
            "max_model_len": 262144,
            "diagnostic_model": MODEL,
            "diagnostic_manifest_sha256": MANIFEST,
        }
        for name in qualifier.DIAGNOSTIC_ARTIFACT_NAMES:
            path = self.root / "m1_84" / name
            if name == "runtime_overlay_identity.json":
                continue
            if name == "runtime_identity.json":
                self._write(path, runtime_identity)
            elif name == "server.log":
                path.write_text("healthy\n", encoding="utf-8")
            else:
                self._write(path, {"qualified": True})
        diagnostic_artifacts = {
            name: hashlib.sha256(
                (self.root / "m1_84" / name).read_bytes()).hexdigest()
            for name in qualifier.DIAGNOSTIC_ARTIFACT_NAMES
        }
        self._write(
            self.root / "m1_84" / "status.json",
            {
                "schema": qualifier.DIAGNOSTIC_SCHEMA,
                "version": 1,
                "qualified": True,
                "gates": {
                    name: 0 for name in qualifier.DIAGNOSTIC_GATE_NAMES
                },
                "runtime_identity": runtime_identity,
                "tool_http_summary": {
                    "streaming_contract_qualified": True,
                    "streaming_equivalence_qualified": True,
                },
                "artifact_sha256": diagnostic_artifacts,
                **authority,
            },
        )

        contract = {
            "source_revision": REVISION,
            "runtime_tree_sha256": TREE,
            "model_manifest_sha256": MANIFEST,
            "model_path": MODEL,
            "tensor_parallel_size": 1,
            "max_model_len": 262144,
            "environment": {"CUDA_VISIBLE_DEVICES": "3"},
        }
        for label, pid, token in (
            ("control", 1001, "a" * 32),
            ("candidate", 2002, "b" * 32),
        ):
            for relative in qualifier.IMAGE_COMPARISON_ARTIFACTS.values():
                if not relative.startswith(f"{label}/"):
                    continue
                path = self.root / "m1_86" / relative
                if relative.endswith("service_contract.json"):
                    self._write(path, contract)
                elif relative.endswith("startup.json"):
                    self._write(path, {
                        "schema": "bi100-http-health-wait-v1",
                        "version": 1,
                        "qualified": True,
                        "reason": "healthy",
                        "attempts": 1,
                    })
                elif relative.endswith("process_group_identity.json"):
                    self._write(path, {
                        "schema": qualifier.SESSION_SCHEMA,
                        "version": 1,
                        "pid": pid,
                        "pgid": pid,
                        "sid": pid,
                        "starttime_ticks": pid * 10,
                        "session_token": token,
                    })
                else:
                    self._write(path, {"qualified": True})
        comparison_artifacts = {
            name: hashlib.sha256(
                (self.root / "m1_86" / relative).read_bytes()).hexdigest()
            for name, relative
            in qualifier.IMAGE_COMPARISON_ARTIFACTS.items()
        }
        self._write(
            self.root / "m1_86" / "comparison.json",
            {
                "schema": qualifier.IMAGE_COMPARISON_SCHEMA,
                "version": 1,
                "qualified": True,
                "artifact_sha256": comparison_artifacts,
                "observed": {"physical_gpu": 3},
                "decision": {
                    "single_gpu_diagnostic_phase_passed": True,
                    "full_model_tp4_required": True,
                    "semantic_quality_required": True,
                    "production_promotion_authorized": False,
                },
            },
        )
        postflight = {
            "schema": qualifier.POSTFLIGHT_SCHEMA,
            "version": 1,
            "qualified": True,
            "gpu_indices": [3],
            "api_server_pids": [],
            "worker_pids": [],
            "gpu_processes": [],
            "scan_errors": [],
        }
        preflight = {
            "schema": "bi100-gpu-preflight-v1",
            "version": 1,
            "ok": True,
            "gpus": [3],
            "results": [{"gpu": 3, "ok": True}],
        }
        self._write(
            self.root / "m1_86" / "final_postflight.json", postflight)
        self._write(
            self.root / "m1_86" / "final_preflight_comparison.json",
            {"qualified": True},
        )
        runner_artifacts = {
            name: hashlib.sha256(
                (self.root / "m1_86" / relative).read_bytes()).hexdigest()
            for name, relative in qualifier.IMAGE_RUNNER_ARTIFACTS.items()
        }
        self._write(
            self.root / "m1_86" / "runner_status.json",
            {
                "schema": qualifier.IMAGE_RUNNER_SCHEMA,
                "version": 1,
                "qualified": True,
                "returncode": 0,
                "source_revision": REVISION,
                "physical_gpu": 3,
                "terminal_stage": "completed",
                "gates": {
                    name: 0 for name in qualifier.IMAGE_RUNNER_GATE_NAMES
                },
                "artifact_sha256": runner_artifacts,
                **authority,
            },
        )
        for name in ("interstage", "final"):
            self._write(self.root / f"{name}_postflight.json", postflight)
            self._write(self.root / f"{name}_preflight.json", preflight)
        for name, pid, token in (
            ("m1_84", 3003, "c" * 32),
            ("m1_86", 4004, "d" * 32),
        ):
            self._write(self.root / f"{name}_child_identity.json", {
                "schema": qualifier.SESSION_SCHEMA,
                "version": 1,
                "pid": pid,
                "pgid": pid,
                "sid": pid,
                "starttime_ticks": pid * 10,
                "session_token": token,
            })
        self._write(self.root / "service_recovery.json", {
            "schema": qualifier.RECOVERY_SCHEMA,
            "version": 1,
            "qualified": True,
            "identity_count": 5,
            "term_grace_s": 60.0,
            "kill_grace_s": 20.0,
            "complete_token_scan_required": True,
            "actions": [
                {
                    "initial_live_count": 0,
                    "initial_escaped_count": 0,
                    "token_scan_error_count": 0,
                    "final_live_count": 0,
                    "term_sent": False,
                    "kill_sent": False,
                    "outcome": "already_quiescent",
                }
                for _ in range(5)
            ],
        })
        for name in (
            "m1_89_overlay_identity",
            "m1_89_runtime_gate",
            "m1_84",
            "interstage_postflight",
            "interstage_preflight",
            "m1_86",
            "final_postflight",
            "final_preflight",
            "fatal_scan",
            "timeout_scan",
            "child_cleanup",
            "service_recovery",
        ):
            (self.root / f"{name}.rc").write_text("0\n", encoding="ascii")

    def _qualify(self, runner_returncode: int = 0) -> dict:
        return qualifier.qualify(
            self.root,
            expected_source_revision=REVISION,
            expected_gpu=3,
            runner_returncode=runner_returncode,
        )

    def test_qualified_queue_binds_identity_and_authority(self) -> None:
        report = self._qualify()
        self.assertTrue(report["qualified"], report["reasons"])
        self.assertEqual(report["schema"], qualifier.SCHEMA)
        self.assertEqual(report["version"], 2)
        self.assertEqual(report["identity"]["runtime_tree_sha256"], TREE)
        self.assertEqual(
            report["identity"]["diagnostic_manifest_sha256"], MANIFEST)
        self.assertFalse(
            report["decision"]["production_promotion_authorized"])
        self.assertFalse(report["semantic_quality_evaluated"])

    def test_overlay_mismatch_fails_closed(self) -> None:
        path = self.root / "m1_86" / "runtime_overlay_identity.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["runtime_tree_sha256"] = "4" * 64
        self._write(path, value)
        report = self._qualify()
        self.assertFalse(report["qualified"])
        self.assertIn(
            "M1-89, M1-84, and M1-86 used different runtime overlays",
            report["reasons"],
        )

    def test_cache_namespace_runtime_failure_fails_before_promotion(
            self) -> None:
        path = self.root / "m1_89_cache_namespace_runtime_gate.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["checks"]["different_palette_isolated"] = False
        value["qualified"] = False
        value["reasons"] = [
            "runtime check failed: different_palette_isolated"]
        self._write(path, value)
        report = self._qualify()
        self.assertFalse(report["qualified"])
        self.assertIn(
            "M1-89 cache namespace runtime gate did not qualify",
            report["reasons"],
        )

    def test_cache_namespace_runtime_path_mismatch_fails_closed(self) -> None:
        path = self.root / "m1_89_cache_namespace_runtime_gate.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["runtime_site_packages"] = "/different/site-packages"
        self._write(path, value)
        report = self._qualify()
        self.assertFalse(report["qualified"])
        self.assertIn(
            "M1-89 gate and overlay runtime paths differ",
            report["reasons"],
        )

    def test_artifact_tamper_fails_closed(self) -> None:
        self._write(
            self.root / "m1_84" / "api_gate.json",
            {"qualified": False},
        )
        report = self._qualify()
        self.assertFalse(report["qualified"])
        self.assertIn(
            "M1-84 artifact binding differs: api_gate.json",
            report["reasons"],
        )

    def test_incomplete_or_extra_gate_set_fails_closed(self) -> None:
        path = self.root / "m1_84" / "status.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["gates"] = {"all": 0}
        self._write(path, value)
        report = self._qualify()
        self.assertFalse(report["qualified"])
        self.assertIn(
            "M1-84 functional diagnostic did not qualify",
            report["reasons"],
        )

    def test_extra_or_path_traversal_artifact_is_rejected(self) -> None:
        path = self.root / "m1_84" / "status.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["artifact_sha256"]["../outside"] = "a" * 64
        self._write(path, value)
        report = self._qualify()
        self.assertFalse(report["qualified"])
        self.assertIn("M1-84 artifact binding is missing", report["reasons"])

    def test_symlinked_artifact_is_rejected_even_with_matching_digest(
            self) -> None:
        artifact = self.root / "m1_84" / "api_gate.json"
        artifact.unlink()
        artifact.symlink_to("/etc/hosts")
        status_path = self.root / "m1_84" / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["artifact_sha256"]["api_gate.json"] = hashlib.sha256(
            artifact.read_bytes()).hexdigest()
        self._write(status_path, status)
        report = self._qualify()
        self.assertFalse(report["qualified"])
        self.assertIn(
            "M1-84 artifact binding differs: api_gate.json",
            report["reasons"],
        )

    def test_emergency_recovery_cannot_qualify_clean_run(self) -> None:
        path = self.root / "service_recovery.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["actions"][0].update({
            "initial_live_count": 1,
            "term_sent": True,
            "outcome": "quiescent",
        })
        self._write(path, value)
        report = self._qualify()
        self.assertFalse(report["qualified"])
        self.assertIn(
            "recorded service recovery was not clean", report["reasons"])

    def test_nonzero_child_or_runner_returncode_fails(self) -> None:
        (self.root / "m1_86.rc").write_text("1\n", encoding="ascii")
        report = self._qualify(runner_returncode=1)
        self.assertFalse(report["qualified"])
        self.assertIn("m1_86 returned 1", report["reasons"])
        self.assertIn("queue primary return code is 1", report["reasons"])

    def test_missing_artifacts_report_reasons_instead_of_raising(self) -> None:
        (self.root / "m1_86" / "comparison.json").unlink()
        report = self._qualify()
        self.assertFalse(report["qualified"])
        self.assertIn(
            "M1-86 comparison is not a confined regular file",
            report["reasons"],
        )

    def test_top_level_symlinked_status_is_rejected(self) -> None:
        status = self.root / "m1_84" / "status.json"
        external = self.root.parent / f"{self.root.name}-status.json"
        external.write_bytes(status.read_bytes())
        status.unlink()
        status.symlink_to(external)
        try:
            report = self._qualify()
        finally:
            external.unlink()
        self.assertFalse(report["qualified"])
        self.assertIn(
            "M1-84 status is not a confined regular file",
            report["reasons"],
        )

    def test_malformed_nested_tool_summary_fails_without_raising(self) -> None:
        path = self.root / "m1_84" / "status.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["tool_http_summary"] = None
        self._write(path, value)
        report = self._qualify()
        self.assertFalse(report["qualified"])
        self.assertIn(
            "M1-84 functional diagnostic did not qualify",
            report["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
