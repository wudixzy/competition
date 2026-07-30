from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_m1_141_short_tp4_screen.py"
SPEC = importlib.util.spec_from_file_location("m1_141_short_tp4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1141ShortTp4RunnerTest(unittest.TestCase):

    def _l2_evidence(self, root: Path) -> tuple[Path, Path, str]:
        extension_sha = "b" * 64
        qualification = {
            "schema": (
                "bi100-fused-prefill-activation-replay-qualification-v1"),
            "version": 1,
            "profile": "qualification",
            "execution_valid": True,
            "stage_qualified": True,
            "invalid_reasons": [],
            "numeric_reasons": [],
            "performance_reasons": [],
            "coverage_reasons": [],
            "report_count": 4,
            "record_count": 36,
            "ranks": [0, 1, 2, 3],
            "capture_source_revision": "a" * 40,
            "candidate_source_revision": "c" * 40,
            "runtime_identity": "runtime-tree",
            "instance": "instance",
            "activation_run_id": "capture-run",
            "bank_manifest_sha256s": [
                f"{index + 1:064x}" for index in range(4)
            ],
            "candidate_extension": {
                "sha256": extension_sha,
                "size_bytes": 247176,
            },
            "median_candidate_speedup": 1.5,
            "minimum_case_speedup": 1.1,
            "contract_sha256": _sha256(
                ROOT / "quality" / "experiment_funnel.v1.json"),
            "numeric_contract_sha256": _sha256(
                ROOT / "quality"
                / "fused_prefill_numeric_adjudication.v1.json"),
            "authorization": {
                "short_tp4_authorized": True,
                "long_context_authorized": False,
                "main_or_yaml_change_authorized": False,
            },
        }
        qualification_path = root / "qualification.json"
        qualification_path.write_text(
            json.dumps(qualification) + "\n", encoding="ascii")
        qualification_path.chmod(0o600)
        status = {
            "schema": "bi100-m1-140-activation-replay-runner-v1",
            "version": 1,
            "qualified": True,
            "returncode": 0,
            "terminal_stage": "complete",
            "profile": "qualification",
            "gpu_count": 4,
            "parallel_rank_replays": 4,
            "capture_source_revision": "a" * 40,
            "candidate_source_revision": "c" * 40,
            "runtime_identity": "runtime-tree",
            "instance": "instance",
            "candidate_extension_sha256": extension_sha,
            "artifact_sha256": {
                "qualification.json": _sha256(qualification_path),
                "timeline_report.json": "d" * 64,
                "preflight_comparison.json": "e" * 64,
                "final_postflight.json": "f" * 64,
            },
            "authorization": {
                "short_tp4_authorized": True,
                "long_context_authorized": False,
                "main_or_yaml_change_authorized": False,
            },
            "privacy": {
                "credentials_recorded": False,
            },
        }
        status_path = root / "runner_status.json"
        status_path.write_text(
            json.dumps(status) + "\n", encoding="ascii")
        status_path.chmod(0o600)
        return qualification_path, status_path, extension_sha

    def test_l2_authorization_binds_artifact_and_lifecycle(self):
        with tempfile.TemporaryDirectory(
                prefix="m1-141-l2-", dir="/tmp") as temporary:
            root = Path(temporary)
            qualification, status, extension_sha = self._l2_evidence(root)
            value = MODULE.validate_l2_authorization(
                qualification,
                status,
                experiment_contract_sha256=_sha256(
                    ROOT / "quality" / "experiment_funnel.v1.json"),
                numeric_contract_sha256=_sha256(
                    ROOT / "quality"
                    / "fused_prefill_numeric_adjudication.v1.json"),
            )
            self.assertEqual(
                value["candidate_extension_sha256"], extension_sha)
            self.assertEqual(value["capture_source_revision"], "a" * 40)

    def test_l2_authorization_rejects_tampered_runner_binding(self):
        with tempfile.TemporaryDirectory(
                prefix="m1-141-l2-", dir="/tmp") as temporary:
            root = Path(temporary)
            qualification, status, _ = self._l2_evidence(root)
            value = json.loads(status.read_text(encoding="ascii"))
            value["artifact_sha256"]["qualification.json"] = "0" * 64
            status.write_text(
                json.dumps(value) + "\n", encoding="ascii")
            status.chmod(0o600)
            with self.assertRaisesRegex(
                    ValueError, "runner identity"):
                MODULE.validate_l2_authorization(
                    qualification,
                    status,
                    experiment_contract_sha256=_sha256(
                        ROOT / "quality"
                        / "experiment_funnel.v1.json"),
                    numeric_contract_sha256=_sha256(
                        ROOT / "quality"
                        / "fused_prefill_numeric_adjudication.v1.json"),
                )

    def test_candidate_service_uses_exact_external_artifact(self):
        args = SimpleNamespace(selector="candidate")
        runner = MODULE.ShortTp4Runner(SimpleNamespace(
            run_root=Path("/tmp/m1-141-unit"),
            instance="instance",
            selector="candidate",
        ))
        runner.args = args
        runner.runtime_site = Path("/tmp/runtime/site-packages")
        runner.runtime_install = Path("/tmp/runtime/install.json")
        runner.run_root = Path("/tmp/m1-141-unit")
        runner.model_path = Path("/tmp/model")
        runner.candidate_extension = Path("/tmp/candidate.so")
        runner.candidate_extension_sha256 = "b" * 64
        environment = runner.service_environment()
        self.assertEqual(
            environment[
                "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION"],
            "/tmp/candidate.so",
        )
        self.assertEqual(
            environment[
                "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256"],
            "b" * 64,
        )

    def test_runner_uses_shared_prompt_identity_and_terminal_gates(self):
        source = SCRIPT.read_text(encoding="ascii")
        request_source = (
            ROOT / "tests" / "short_tp4_funnel_service.py"
        ).read_text(encoding="ascii")
        self.assertIn("--prompt-set-id", source)
        self.assertIn('--repetitions", "3"', source)
        self.assertIn("self.args.pair_id", source)
        self.assertIn("self.run_postconditions(primary_error)", source)
        self.assertIn("args.prompt_set_id", request_source)
        self.assertIn('"repetitions": args.repetitions', request_source)
        self.assertIn('"prompt_sha256": prompt_sha256', request_source)

    def test_missing_failure_artifact_has_null_digest(self):
        with tempfile.TemporaryDirectory(
                prefix="m1-141-status-", dir="/tmp") as temporary:
            missing = Path(temporary) / "measurement.json"
            self.assertIsNone(MODULE._sha256_if_file(missing))
            missing.write_text("{}\n", encoding="ascii")
            self.assertEqual(
                MODULE._sha256_if_file(missing),
                _sha256(missing),
            )


if __name__ == "__main__":
    unittest.main()
