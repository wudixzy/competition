from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

SPEC = importlib.util.spec_from_file_location(
    "compare_admission64_quality_service_ab",
    TESTS / "compare_admission64_quality_service_ab.py",
)
COMPARE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(COMPARE)

RUNNER = ROOT / "scripts/run_m1_85_admission64_quality_ab.sh"
QUALITY_RUNNER = ROOT / "scripts/run_quality_service_gate.sh"
REVISION = "a" * 40
INSTANCE = "private-tp4"
BRANCH = "test/M1-85-admission64-quality-ab-20260728"
FILE_SHA256S = {
    "control": {
        "runtime_contract": "1" * 64,
        "quality_report": "2" * 64,
        "agent_workload": "3" * 64,
        "api_4xx_attribution": "4" * 64,
        "process_group_identity": "9" * 64,
        "service_recovery": "a" * 64,
        "service_recovery_clean": "c" * 64,
    },
    "candidate": {
        "runtime_contract": "5" * 64,
        "quality_report": "6" * 64,
        "agent_workload": "7" * 64,
        "api_4xx_attribution": "8" * 64,
        "process_group_identity": "0" * 64,
        "service_recovery": "b" * 64,
        "service_recovery_clean": "d" * 64,
    },
}


def label(policy: str) -> str:
    return {
        "fine32": "m1-85-control-fine32",
        "admission64": "m1-85-candidate-admission64",
    }[policy]


def contract(policy: str) -> dict:
    runtime = COMPARE.runtime_contract
    return {
        "schema": "bi100-quality-runtime-contract-v1",
        "version": 1,
        "source_revision": REVISION,
        "runtime_identity": "unit-runtime",
        "runtime_overlay_sha256": "b" * 64,
        "instance": INSTANCE,
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": "/model",
        "tokenizer_path": "/model",
        "served_model_name": "llm",
        "base_image": runtime.BASE_IMAGE,
        "command": runtime.service_command("/model"),
        "environment": runtime.service_environment(
            "/runtime/site-packages",
            gdn_cache_policy=policy,
            gdn_restore_mode="direct",
            fused_prefill="0",
            kv_eviction_policy="lru",
            kernel_profile="submission",
        ),
        "cache_trace_enabled": True,
        "optimization_label": label(policy),
    }


def status(policy: str) -> dict:
    return {
        "schema": COMPARE.STATUS_SCHEMA,
        "version": 2,
        "suite": "functional",
        "optimization": {
            **COMPARE.EXPECTED_COMMON_OPTIMIZATION,
            "gdn_cache_policy": policy,
        },
        "label": label(policy),
        "instance": INSTANCE,
        "overall_rc": 0,
        "source_revision": REVISION,
        "source_branch": BRANCH,
        "gates": {
            name: 0 for name in COMPARE.EXPECTED_STATUS_GATES
        },
        "privacy": {
            "raw_service_log_outside_repository": True,
            "contains_credentials": False,
        },
        "artifacts": {
            "runtime_contract_sha256": FILE_SHA256S[policy_key(policy)][
                "runtime_contract"],
            "quality_report_sha256": FILE_SHA256S[policy_key(policy)][
                "quality_report"],
            "agent_workload_sha256": FILE_SHA256S[policy_key(policy)][
                "agent_workload"],
            "api_4xx_attribution_sha256": FILE_SHA256S[policy_key(policy)][
                "api_4xx_attribution"],
            "process_group_identity_sha256": FILE_SHA256S[
                policy_key(policy)]["process_group_identity"],
            "service_recovery_sha256": FILE_SHA256S[
                policy_key(policy)]["service_recovery"],
            "service_recovery_clean_sha256": FILE_SHA256S[
                policy_key(policy)]["service_recovery_clean"],
        },
    }


def policy_key(policy: str) -> str:
    return "control" if policy == "fine32" else "candidate"


def quality_comparison() -> dict:
    return {
        "schema": COMPARE.quality_compare.COMPARISON_SCHEMA,
        "version": 1,
        "qualified": True,
        "quality_non_regression_authorized": True,
        "overall_promotion_authorized": False,
        "reasons": [],
        "summary": {
            "compared_cases": COMPARE.quality_compare.EXPECTED_CASES,
            "qualified_cases": COMPARE.quality_compare.EXPECTED_CASES,
            "failed_cases": 0,
        },
        "cases": [
            {
                "id": case_id,
                "qualified": True,
                "reasons": [],
            }
            for case_id in COMPARE.EXPECTED_QUALITY_IDS
        ],
        "privacy": COMPARE.EXPECTED_QUALITY_COMPARISON_PRIVACY.copy(),
        "inputs": {
            "baseline_file_sha256": FILE_SHA256S["control"][
                "quality_report"],
            "candidate_file_sha256": FILE_SHA256S["candidate"][
                "quality_report"],
        },
    }


def agent_comparison() -> dict:
    return {
        "schema": COMPARE.agent_compare.SCHEMA,
        "version": 1,
        "qualified": True,
        "agent_quality_non_regression_authorized": True,
        "overall_promotion_authorized": False,
        "reasons": [],
        "summary": {
            "compared_cases": COMPARE.EXPECTED_AGENT_CASES,
            "qualified_cases": COMPARE.EXPECTED_AGENT_CASES,
        },
        "cases": [
            {
                "id": case_id,
                "qualified": True,
                "reasons": [],
            }
            for case_id in COMPARE.EXPECTED_AGENT_IDS
        ],
        "inputs": {
            "baseline_file_sha256": FILE_SHA256S["control"][
                "agent_workload"],
            "candidate_file_sha256": FILE_SHA256S["candidate"][
                "agent_workload"],
        },
    }


def api_4xx_report() -> dict:
    return {
        "schema": COMPARE.api_4xx.REPORT_SCHEMA,
        "version": COMPARE.api_4xx.REPORT_VERSION,
        "complete": True,
        "classified": True,
        "qualified": True,
        "chat_4xx_access_count": 8,
        "attributed_count": 8,
        "attribution_delta": 0,
        "malformed_marker_count": 0,
        "by_access_code": {"400": 8},
        "by_attributed_code": {"400": 8},
        "by_endpoint": {"chat": 1, "request_validation": 7},
        "by_reason": {
            "empty_messages": 1,
            "request_validation_sampling": 7,
        },
        "request_shapes": [],
        "privacy": COMPARE.EXPECTED_4XX_PRIVACY.copy(),
    }


def process_identity(policy: str) -> dict:
    pid = 100 if policy == "fine32" else 200
    return {
        "schema": "bi100-process-session-v1",
        "version": 1,
        "pid": pid,
        "pgid": pid,
        "sid": pid,
        "starttime_ticks": pid * 10,
        "session_token": "1" * 32 if policy == "fine32" else "2" * 32,
    }


def compare_fixture() -> dict:
    return COMPARE.compare(
        control_status=status("fine32"),
        candidate_status=status("admission64"),
        control_contract=contract("fine32"),
        candidate_contract=contract("admission64"),
        control_4xx=api_4xx_report(),
        candidate_4xx=api_4xx_report(),
        control_process_identity=process_identity("fine32"),
        candidate_process_identity=process_identity("admission64"),
        quality_comparison=quality_comparison(),
        agent_comparison=agent_comparison(),
        file_sha256s=copy.deepcopy(FILE_SHA256S),
    )


def write_json(path: Path, value: dict) -> str:
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class Admission64QualityServiceAbTest(unittest.TestCase):

    def test_exact_policy_only_quality_ab_qualifies(self):
        result = compare_fixture()
        self.assertTrue(result["qualified"], result)
        self.assertTrue(
            result["admission64_quality_non_regression_authorized"])
        self.assertTrue(result["policy_only_runtime_delta_attested"])
        self.assertFalse(result["performance_authorized"])
        self.assertFalse(result["default_policy_change_authorized"])
        self.assertFalse(result["production_promotion_authorized"])

    def test_failed_child_gate_rejects_evidence(self):
        control = status("fine32")
        control["gates"]["api_4xx_attribution"] = 1
        result = COMPARE.compare(
            control_status=control,
            candidate_status=status("admission64"),
            control_contract=contract("fine32"),
            candidate_contract=contract("admission64"),
            control_4xx=api_4xx_report(),
            candidate_4xx=api_4xx_report(),
            control_process_identity=process_identity("fine32"),
            candidate_process_identity=process_identity("admission64"),
            quality_comparison=quality_comparison(),
            agent_comparison=agent_comparison(),
            file_sha256s=copy.deepcopy(FILE_SHA256S),
        )
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "api_4xx_attribution" in reason for reason in result["reasons"]))

    def test_artifact_rebinding_is_rejected(self):
        candidate = status("admission64")
        candidate["artifacts"]["quality_report_sha256"] = "9" * 64
        result = COMPARE.compare(
            control_status=status("fine32"),
            candidate_status=candidate,
            control_contract=contract("fine32"),
            candidate_contract=contract("admission64"),
            control_4xx=api_4xx_report(),
            candidate_4xx=api_4xx_report(),
            control_process_identity=process_identity("fine32"),
            candidate_process_identity=process_identity("admission64"),
            quality_comparison=quality_comparison(),
            agent_comparison=agent_comparison(),
            file_sha256s=copy.deepcopy(FILE_SHA256S),
        )
        self.assertFalse(result["qualified"])
        self.assertIn(
            "candidate: status artifact bindings differ", result["reasons"])

    def test_extra_environment_delta_is_rejected(self):
        candidate_contract = contract("admission64")
        candidate_contract["environment"]["BI100_UNDECLARED"] = "1"
        result = COMPARE.compare(
            control_status=status("fine32"),
            candidate_status=status("admission64"),
            control_contract=contract("fine32"),
            candidate_contract=candidate_contract,
            control_4xx=api_4xx_report(),
            candidate_4xx=api_4xx_report(),
            control_process_identity=process_identity("fine32"),
            candidate_process_identity=process_identity("admission64"),
            quality_comparison=quality_comparison(),
            agent_comparison=agent_comparison(),
            file_sha256s=copy.deepcopy(FILE_SHA256S),
        )
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "environment" in reason for reason in result["reasons"]))

    def test_4xx_shape_or_reason_change_is_rejected(self):
        candidate_4xx = api_4xx_report()
        candidate_4xx["by_reason"] = {
            "empty_messages": 1,
            "request_validation_sampling": 6,
            "unclassified_chat_error": 1,
        }
        result = COMPARE.compare(
            control_status=status("fine32"),
            candidate_status=status("admission64"),
            control_contract=contract("fine32"),
            candidate_contract=contract("admission64"),
            control_4xx=api_4xx_report(),
            candidate_4xx=candidate_4xx,
            control_process_identity=process_identity("fine32"),
            candidate_process_identity=process_identity("admission64"),
            quality_comparison=quality_comparison(),
            agent_comparison=agent_comparison(),
            file_sha256s=copy.deepcopy(FILE_SHA256S),
        )
        self.assertFalse(result["qualified"])
        self.assertIn(
            "A/B 4xx attribution or request shapes differ", result["reasons"])

    def test_incomplete_component_comparison_is_rejected(self):
        quality = quality_comparison()
        quality["cases"].pop()
        result = COMPARE.compare(
            control_status=status("fine32"),
            candidate_status=status("admission64"),
            control_contract=contract("fine32"),
            candidate_contract=contract("admission64"),
            control_4xx=api_4xx_report(),
            candidate_4xx=api_4xx_report(),
            control_process_identity=process_identity("fine32"),
            candidate_process_identity=process_identity("admission64"),
            quality_comparison=quality,
            agent_comparison=agent_comparison(),
            file_sha256s=copy.deepcopy(FILE_SHA256S),
        )
        self.assertFalse(result["qualified"])
        self.assertIn(
            "quality: qualified case evidence is incomplete",
            result["reasons"],
        )

    def test_malformed_process_identity_is_rejected(self):
        candidate_identity = process_identity("admission64")
        candidate_identity["starttime_ticks"] = 0
        result = COMPARE.compare(
            control_status=status("fine32"),
            candidate_status=status("admission64"),
            control_contract=contract("fine32"),
            candidate_contract=contract("admission64"),
            control_4xx=api_4xx_report(),
            candidate_4xx=api_4xx_report(),
            control_process_identity=process_identity("fine32"),
            candidate_process_identity=candidate_identity,
            quality_comparison=quality_comparison(),
            agent_comparison=agent_comparison(),
            file_sha256s=copy.deepcopy(FILE_SHA256S),
        )
        self.assertFalse(result["qualified"])
        self.assertIn(
            "candidate: process identity is not attested",
            result["reasons"],
        )

    def test_reused_process_session_token_is_rejected(self):
        candidate_identity = process_identity("admission64")
        candidate_identity["session_token"] = process_identity(
            "fine32")["session_token"]
        result = COMPARE.compare(
            control_status=status("fine32"),
            candidate_status=status("admission64"),
            control_contract=contract("fine32"),
            candidate_contract=contract("admission64"),
            control_4xx=api_4xx_report(),
            candidate_4xx=api_4xx_report(),
            control_process_identity=process_identity("fine32"),
            candidate_process_identity=candidate_identity,
            quality_comparison=quality_comparison(),
            agent_comparison=agent_comparison(),
            file_sha256s=copy.deepcopy(FILE_SHA256S),
        )
        self.assertFalse(result["qualified"])
        self.assertIn(
            "A/B process session tokens must differ", result["reasons"])

    def test_cli_binds_exact_arm_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm_hashes = {}
            for arm, policy in (
                    ("control", "fine32"),
                    ("candidate", "admission64")):
                arm_root = root / arm
                arm_root.mkdir()
                arm_hashes[arm] = {
                    "runtime_contract": write_json(
                        arm_root / "runtime_contract.json",
                        contract(policy),
                    ),
                    "quality_report": write_json(
                        arm_root / "quality_report.json", {}),
                    "agent_workload": write_json(
                        arm_root / "agent_workload.json", {}),
                    "api_4xx_attribution": write_json(
                        arm_root / "api_4xx_attribution.json",
                        api_4xx_report(),
                    ),
                    "process_group_identity": write_json(
                        arm_root / "process_group_identity.json",
                        {
                            "schema": "bi100-process-session-v1",
                            "version": 1,
                            "pid": 100 if arm == "control" else 200,
                            "pgid": 100 if arm == "control" else 200,
                            "sid": 100 if arm == "control" else 200,
                            "starttime_ticks": (
                                1000 if arm == "control" else 2000),
                            "session_token": (
                                "1" * 32 if arm == "control" else "2" * 32),
                        },
                    ),
                    "service_recovery": write_json(
                        arm_root / "service_recovery.json", {}),
                    "service_recovery_clean": write_json(
                        arm_root / "service_recovery_clean.json", {}),
                }
                arm_status = status(policy)
                arm_status["artifacts"] = {
                    "runtime_contract_sha256": arm_hashes[arm][
                        "runtime_contract"],
                    "quality_report_sha256": arm_hashes[arm][
                        "quality_report"],
                    "agent_workload_sha256": arm_hashes[arm][
                        "agent_workload"],
                    "api_4xx_attribution_sha256": arm_hashes[arm][
                        "api_4xx_attribution"],
                    "process_group_identity_sha256": arm_hashes[arm][
                        "process_group_identity"],
                    "service_recovery_sha256": arm_hashes[arm][
                        "service_recovery"],
                    "service_recovery_clean_sha256": arm_hashes[arm][
                        "service_recovery_clean"],
                }
                write_json(arm_root / "status.json", arm_status)

            quality = quality_comparison()
            quality["inputs"] = {
                "baseline_file_sha256": arm_hashes["control"][
                    "quality_report"],
                "candidate_file_sha256": arm_hashes["candidate"][
                    "quality_report"],
            }
            agent = agent_comparison()
            agent["inputs"] = {
                "baseline_file_sha256": arm_hashes["control"][
                    "agent_workload"],
                "candidate_file_sha256": arm_hashes["candidate"][
                    "agent_workload"],
            }
            quality_path = root / "quality-comparison.json"
            agent_path = root / "agent-comparison.json"
            output_path = root / "aggregate.json"
            write_json(quality_path, quality)
            write_json(agent_path, agent)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(TESTS / "compare_admission64_quality_service_ab.py"),
                    "--control-root", str(root / "control"),
                    "--candidate-root", str(root / "candidate"),
                    "--quality-comparison", str(quality_path),
                    "--agent-comparison", str(agent_path),
                    "--out", str(output_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            aggregate = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(aggregate["qualified"], aggregate)
            self.assertIn("inputs", aggregate)
            self.assertEqual(
                aggregate["inputs"][
                    "control_process_group_identity_sha256"],
                arm_hashes["control"]["process_group_identity"],
            )


class Admission64QualityRunnerStaticTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.quality_source = QUALITY_RUNNER.read_text(encoding="utf-8")
        cls.comparator_source = (
            TESTS / "compare_admission64_quality_service_ab.py"
        ).read_text(encoding="utf-8")

    def test_fixed_order_changes_only_cache_policy(self):
        control = self.source.index(
            "run_arm control fine32 m1-85-control-fine32")
        candidate = self.source.index(
            "run_arm candidate admission64 m1-85-candidate-admission64")
        self.assertLess(control, candidate)
        self.assertIn(
            '"$ROOT/scripts/run_quality_service_gate.sh" \\',
            self.source,
        )
        self.assertIn(
            'functional "$policy" "$restore_mode" "$fused_prefill" lru',
            self.source,
        )
        self.assertIn("local restore_mode=direct", self.source)
        self.assertIn("local fused_prefill=0", self.source)
        self.assertIn(
            "BI100_QUALITY_KERNEL_PROFILE=submission", self.source)
        self.assertNotIn("computility-run.yaml", self.source)

    def test_both_quality_dimensions_and_4xx_are_aggregated(self):
        self.assertIn("compare_quality_gate_reports.py", self.source)
        self.assertIn("compare_agent_workload_reports.py", self.source)
        self.assertIn(
            "compare_admission64_quality_service_ab.py", self.source)
        self.assertIn("api_4xx_attribution", self.comparator_source)

    def test_outer_cleanup_is_graceful_and_fail_closed(self):
        self.assertIn("source \"$ROOT/scripts/lib/process_group.sh\"",
                      self.source)
        self.assertIn("bi100_stop_process_group", self.source)
        self.assertIn("CHILD_TERM_GRACE_S=60", self.source)
        self.assertIn("CHILD_KILL_GRACE_S=20", self.source)
        self.assertNotIn("CHILD_TERM_GRACE_S=900", self.source)
        self.assertIn("exec_bi100_session.py", self.source)
        self.assertIn("ACTIVE_CHILD_STARTTIME", self.source)
        self.assertIn("ACTIVE_CHILD_SESSION_TOKEN", self.source)
        self.assertIn("cleanup_recorded_bi100_sessions.py", self.source)
        self.assertIn("qualify_recorded_session_cleanup.py", self.source)
        self.assertIn("orchestrator_recovery_clean", self.source)
        self.assertIn(
            '"bi100-admission64-quality-ab-runner-v2"',
            self.source,
        )
        self.assertIn('"control_child_identity",', self.source)
        self.assertIn('"control_service_identity",', self.source)
        self.assertIn('"candidate_child_identity",', self.source)
        self.assertIn('"candidate_service_identity",', self.source)
        self.assertIn(
            '"$RUN_ROOT/control/process_group_identity.json"',
            self.source,
        )
        self.assertIn(
            '"$RUN_ROOT/candidate/process_group_identity.json"',
            self.source,
        )
        self.assertIn('artifacts[f"{name}_sha256"]', self.source)
        self.assertIn("trap finish EXIT", self.source)
        self.assertIn("trap '' TERM INT", self.source)
        self.assertIn("service_postflight_gate.py", self.source)
        self.assertIn("--gpus 0,1,2,3", self.source)
        self.assertIn("bi100_preflight.py", self.source)
        self.assertIn("scan_orchestrator_fatal_logs", self.source)
        self.assertIn("scan_orchestrator_timeouts", self.source)
        self.assertIn("-name '*.stdout'", self.source)
        self.assertIn("-name '*.stderr'", self.source)
        self.assertIn("-name '*.rc'", self.source)
        self.assertIn("124|137|143", self.source)
        self.assertNotIn("pkill", self.source)

    def test_inner_service_is_attested_and_gracefully_reaped(self):
        self.assertIn("exec_bi100_session.py", self.quality_source)
        self.assertIn(
            '"$RUN_ROOT/process_group_identity.json"', self.quality_source)
        self.assertIn(
            '"$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            self.quality_source,
        )
        self.assertIn("ACTIVE_STARTTIME", self.quality_source)
        self.assertIn("ACTIVE_SESSION_TOKEN", self.quality_source)
        self.assertIn("active_pid_is_same", self.quality_source)
        self.assertIn("wait \"$ACTIVE_PID\"", self.quality_source)
        self.assertIn("trap '' TERM INT", self.quality_source)
        self.assertIn(
            "cleanup_recorded_bi100_sessions.py", self.quality_source)
        self.assertIn(
            "qualify_recorded_session_cleanup.py", self.quality_source)
        self.assertNotIn(
            'setsid "$ROOT/launch_service"', self.quality_source)
        self.assertNotIn("pkill", self.quality_source)

    def test_inner_status_binds_process_identity_and_timeout_scans(self):
        self.assertIn(
            '"schema": "bi100-quality-service-gate-status-v2"',
            self.quality_source,
        )
        self.assertIn(
            '"process_group": read_rc("process_group.rc")',
            self.quality_source,
        )
        self.assertIn(
            '"process_group_identity_sha256"', self.quality_source)
        self.assertIn(
            '"service_recovery": read_rc("service_recovery.rc")',
            self.quality_source,
        )
        self.assertIn(
            '"service_recovery_clean": read_rc("service_recovery_clean.rc")',
            self.quality_source,
        )
        self.assertIn('"service_recovery_sha256"', self.quality_source)
        self.assertIn(
            '"service_recovery_clean_sha256"', self.quality_source)
        self.assertIn("-name '*.rc'", self.quality_source)
        self.assertIn("124|137|143", self.quality_source)


if __name__ == "__main__":
    unittest.main()
