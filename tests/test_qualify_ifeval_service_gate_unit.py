from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/qualify_ifeval_service_gate.py"
SPEC = importlib.util.spec_from_file_location("qualify_ifeval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
RUNTIME_REVISION = "a" * 40
EVALUATOR_REVISION = "b" * 40


class QualifyIFEvalServiceGateTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.install = self.root / "install.json"
        self._write_fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def _write(self, name: str, value: object) -> None:
        (self.run / name).write_text(
            json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

    def _write_fixture(self) -> None:
        optimization = {
            "gdn_cache_policy": "fine32",
            "gdn_restore_mode": "direct",
            "fused_prefill": "0",
            "kv_eviction_policy": "lru",
        }
        report_optimization = dict(optimization)
        report_optimization["fused_prefill"] = False
        for gate in MODULE.EXPECTED_GATES:
            (self.run / f"{gate}.rc").write_text("0\n", encoding="ascii")
        (self.run / "overall.rc").write_text("0\n", encoding="ascii")
        self._write("runtime_identity.json", {"qualified": True})
        self._write("runtime_contract.json", {"schema": "runtime"})
        self._write("startup_contract.json", {"qualified": True})
        self._write("preflight_before.json", {"qualified": True})
        self._write("preflight_after.json", {"qualified": True})
        self._write("preflight_comparison.json", {"qualified": True})
        (self.run / "fatal_scan.txt").write_bytes(b"")
        contract_sha = MODULE.sha256(self.run / "runtime_contract.json")
        self._write("ifeval_report.json", {
            "schema": MODULE.REPORT_SCHEMA,
            "version": 1,
            "run_id_sha256": "c" * 64,
            "qualified": True,
            "quality_run_eligible_for_baseline": True,
            "promotion_authorized": False,
            "manifest": {
                "sha256": MODULE.EXPECTED_MANIFEST_SHA256,
                "full_selection": True,
                "selected_keys": list(range(64)),
            },
            "runtime": {
                "source_revision": RUNTIME_REVISION,
                "gpu_count": 4,
                "tensor_parallel_size": 4,
                "max_model_len": 262144,
                "optimization": report_optimization,
            },
            "runtime_contract": {"file_sha256": contract_sha},
            "summary": {"prompt_total": 64, "instruction_total": 100},
            "transport": {"selected": 64, "completed": 64, "errors": 0},
            "cases": [{"key": key, "status": "pass"} for key in range(64)],
            "privacy": {
                "contains_credentials": False,
                "contains_raw_prompts": False,
                "contains_raw_model_outputs": False,
                "contains_reasoning_text": False,
                "checkpoint_deleted": True,
            },
        })
        report_sha = MODULE.sha256(self.run / "ifeval_report.json")
        self._write("ifeval_progress.json", {
            "schema": "bi100-ifeval-progress-v1",
            "version": 1,
            "run_id_sha256": "c" * 64,
            "selected": 64,
            "attempted": 64,
            "successful": 64,
            "errors": 0,
            "last_ordinal": 64,
            "complete": True,
            "report_sha256": report_sha,
            "failures": [],
            "privacy": {
                "contains_credentials": False,
                "contains_raw_prompts": False,
                "contains_raw_model_outputs": False,
                "contains_reasoning_text": False,
            },
        })
        artifacts = {
            name: MODULE.sha256(self.run / name)
            for name in MODULE.EXPECTED_ARTIFACTS
        }
        self._write("status.json", {
            "schema": MODULE.STATUS_SCHEMA,
            "version": 1,
            "overall_rc": 0,
            "runtime_source_revision": RUNTIME_REVISION,
            "evaluator_source_revision": EVALUATOR_REVISION,
            "optimization": optimization,
            "gates": {gate: 0 for gate in MODULE.EXPECTED_GATES},
            "artifacts": artifacts,
            "privacy": {
                "raw_service_log_outside_repository": True,
                "raw_checkpoint_absent_after_lifecycle": True,
                "contains_credentials": False,
            },
        })
        self.install.write_text(json.dumps({
            "schema": MODULE.INSTALL_SCHEMA,
            "version": 1,
            "qualified": True,
            "manifest_sha256": MODULE.EXPECTED_MANIFEST_SHA256,
            "python": "3.10.12",
            "system_site_packages_modified": False,
        }) + "\n", encoding="utf-8")

    def qualify(self):
        return MODULE.qualify(
            self.run, self.install, RUNTIME_REVISION, EVALUATOR_REVISION)

    def test_valid_service_run_qualifies(self):
        result = self.qualify()
        self.assertTrue(result["qualified"], result["reasons"])
        self.assertFalse(result["promotion_authorized"])
        self.assertEqual(result["transport"]["completed"], 64)

    def test_failed_gate_and_status_drift_fail_closed(self):
        (self.run / "startup.rc").write_text("1\n", encoding="ascii")
        result = self.qualify()
        self.assertFalse(result["qualified"])
        self.assertTrue(any("startup" in reason for reason in result["reasons"]))

    def test_retained_raw_checkpoint_fails_closed(self):
        (self.run / "ifeval.checkpoint.json").write_text(
            "private", encoding="utf-8")
        result = self.qualify()
        self.assertFalse(result["qualified"])
        self.assertIn("raw IFEval checkpoint was retained", result["reasons"])

    def test_runtime_or_artifact_drift_fails_closed(self):
        self._write("runtime_contract.json", {"schema": "drifted"})
        result = self.qualify()
        self.assertFalse(result["qualified"])
        self.assertTrue(any("runtime_contract.json" in reason
                            for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
