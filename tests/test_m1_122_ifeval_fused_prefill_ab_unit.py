from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
COMPARATOR = TESTS / "compare_m1_122_ifeval_service_ab.py"
SERVICE_RUNNER = ROOT / "scripts/run_quality_service_gate.sh"
OUTER_RUNNER = ROOT / "scripts/run_m1_85_admission64_quality_ab.sh"
WRAPPER = ROOT / "scripts/run_m1_122_ifeval_fused_prefill_ab.sh"
IFEVAL_API = TESTS / "ifeval_quality_api.py"


def load_comparator():
    sys.path.insert(0, str(TESTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "compare_m1_122_ifeval_service_ab_unit",
            COMPARATOR,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load M1-122 comparator")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


M = load_comparator()
SHA = "a" * 64
SOURCE = "b" * 40
INSTANCE = "ssh-73ca29ba"


def contract(label: str, fused: str) -> dict:
    runtime = M.service.runtime_contract
    model = "/model"
    return {
        "schema": "bi100-quality-runtime-contract-v1",
        "version": 1,
        "source_revision": SOURCE,
        "runtime_identity": "bare-host-overlay-v1:" + "c" * 20,
        "runtime_overlay_sha256": "c" * 64,
        "instance": INSTANCE,
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": model,
        "tokenizer_path": model,
        "served_model_name": "llm",
        "base_image": runtime.BASE_IMAGE,
        "command": runtime.service_command(model),
        "environment": runtime.service_environment(
            "/overlay/site-packages",
            gdn_cache_policy="admission64",
            gdn_restore_mode="hybrid64",
            fused_prefill=fused,
            kv_eviction_policy="lru",
            kernel_profile="submission",
        ),
        "cache_trace_enabled": True,
        "optimization_label": label,
    }


def report(label: str, fused: str, output: str = "d" * 64) -> dict:
    runtime_contract = contract(label, fused)
    counts = {"total": 64, "strict_passed": 64, "loose_passed": 64}
    return {
        "schema": "bi100-ifeval-result-v1",
        "version": 1,
        "qualified": True,
        "quality_run_eligible_for_baseline": True,
        "promotion_authorized": False,
        "run_id_sha256": "e" * 64,
        "manifest": {
            "sha256": M.ifeval.EXPECTED_MANIFEST_SHA256,
            "full_selection": True,
            "selected_keys": list(range(64)),
        },
        "runtime": {
            "source_revision": SOURCE,
            "runtime_identity": runtime_contract["runtime_identity"],
            "runtime_overlay_sha256": runtime_contract[
                "runtime_overlay_sha256"],
            "runtime_contract_sha256": "f" * 64,
            "instance": INSTANCE,
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "model": "llm",
            "model_path": "/model",
            "tokenizer_path": "/model",
            "optimization": {
                "gdn_cache_policy": "admission64",
                "gdn_restore_mode": "hybrid64",
                "fused_prefill": fused == "1",
                "kv_eviction_policy": "lru",
            },
        },
        "runtime_contract": {
            "sha256": "f" * 64,
            "file_sha256": SHA,
            "contract": runtime_contract,
        },
        "request_conversion": {
            "max_tokens": 8192,
            "temperature": 0,
            "seed": 20260725,
            "stream": False,
        },
        "evaluator": {
            "revision": "1" * 40,
            "strict_and_loose_rules_unmodified": True,
        },
        "summary": {
            "prompt_total": 64,
            "instruction_total": 64,
            "strict_prompt_passed": 64,
            "loose_prompt_passed": 64,
            "strict_instruction_passed": 64,
            "loose_instruction_passed": 64,
            "by_instruction_id": {
                "keywords:existence": dict(counts),
            },
            "by_family": {
                "keywords": dict(counts),
            },
        },
        "transport": {"selected": 64, "completed": 64, "errors": 0},
        "cases": [
            {
                "key": key,
                "status": "pass",
                "instruction_id_list": ["keywords:existence"],
                "strict": [True],
                "loose": [True],
                "semantic_output_sha256": output,
            }
            for key in range(64)
        ],
        "privacy": {
            "contains_credentials": False,
            "contains_raw_prompts": False,
            "contains_raw_model_outputs": False,
            "contains_reasoning_text": False,
            "checkpoint_deleted": True,
        },
    }


def progress(report_value: dict, report_sha256: str) -> dict:
    return {
        "schema": "bi100-ifeval-progress-v1",
        "version": 1,
        "run_id_sha256": report_value["run_id_sha256"],
        "selected": 64,
        "attempted": 64,
        "successful": 64,
        "errors": 0,
        "last_ordinal": 64,
        "complete": True,
        "report_sha256": report_sha256,
        "failures": [],
        "privacy": {
            "contains_credentials": False,
            "contains_raw_prompts": False,
            "contains_raw_model_outputs": False,
            "contains_reasoning_text": False,
        },
    }


def zero_4xx() -> dict:
    return {
        "schema": M.api_4xx.REPORT_SCHEMA,
        "version": M.api_4xx.REPORT_VERSION,
        "complete": True,
        "classified": True,
        "qualified": True,
        "chat_4xx_access_count": 0,
        "attributed_count": 0,
        "attribution_delta": 0,
        "malformed_marker_count": 0,
        "by_access_code": {},
        "by_attributed_code": {},
        "by_endpoint": {},
        "by_reason": {},
        "by_validation_field": {},
        "by_validation_type": {},
        "request_shapes": [],
        "privacy": M.service.EXPECTED_4XX_PRIVACY,
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


def hashes() -> dict[str, str]:
    return {name: SHA for name in M.EXPECTED_ARTIFACTS}


def status(label: str, fused: str, file_sha256s: dict[str, str]) -> dict:
    return {
        "schema": M.service.STATUS_SCHEMA,
        "version": 2,
        "suite": "ifeval",
        "optimization": M._optimization(fused),
        "label": label,
        "instance": INSTANCE,
        "overall_rc": 0,
        "source_revision": SOURCE,
        "source_branch": "exp/M1-122",
        "gates": {name: 0 for name in M.EXPECTED_GATES},
        "artifacts": {
            f"{name}_sha256": file_sha256s[name]
            for name in sorted(M.EXPECTED_ARTIFACTS)
        },
        "privacy": M.EXPECTED_PRIVACY,
    }


def comparison(
    control: dict,
    candidate: dict,
    *,
    exact: bool,
) -> dict:
    reasons = M.ifeval.comparison_reasons(
        control, candidate, {"fused_prefill"}, exact)
    return {
        "schema": M.ifeval.COMPARISON_SCHEMA,
        "version": 1,
        "qualified": not reasons,
        "promotion_authorized": False,
        "baseline_sha256": SHA,
        "candidate_sha256": SHA,
        "allowed_switches": ["fused_prefill"],
        "require_exact_output": exact,
        "reasons": reasons,
        "score_delta": {
            name: candidate["summary"][name] - control["summary"][name]
            for name in (
                "strict_prompt_passed",
                "loose_prompt_passed",
                "strict_instruction_passed",
                "loose_instruction_passed",
            )
        },
    }


def paired_comparison(control: dict, candidate: dict) -> dict:
    quality_contract = json.loads(
        (ROOT / "quality/layered_quality_gate.v1.json").read_text(
            encoding="utf-8"))
    value = M.paired_ifeval.compare(
        control,
        candidate,
        quality_contract,
        allowed_switches={"fused_prefill"},
    )
    value.update({
        "baseline_sha256": SHA,
        "candidate_sha256": SHA,
        "contract_sha256": M.EXPECTED_LAYERED_CONTRACT_SHA256,
    })
    return value


def invoke(
    *,
    output_drift: bool = False,
    single_paired_regression: bool = False,
    paired_regression: bool = False,
    failed_gate: bool = False,
) -> dict:
    control_report = report("m1-122-control-fused-off", "0")
    candidate_report = report(
        "m1-122-candidate-fused-on",
        "1",
        output="0" * 64 if output_drift else "d" * 64,
    )
    if single_paired_regression:
        candidate_report["cases"][0]["strict"] = [False]
        candidate_report["cases"][0]["semantic_output_sha256"] = "0" * 64
        candidate_report["summary"]["strict_prompt_passed"] -= 1
        candidate_report["summary"]["strict_instruction_passed"] -= 1
        candidate_report["summary"]["by_instruction_id"][
            "keywords:existence"]["strict_passed"] -= 1
        candidate_report["summary"]["by_family"][
            "keywords"]["strict_passed"] -= 1
    if paired_regression:
        for case in candidate_report["cases"][:10]:
            case["strict"] = [False]
            case["semantic_output_sha256"] = "0" * 64
        candidate_report["summary"]["strict_prompt_passed"] -= 10
        candidate_report["summary"]["strict_instruction_passed"] -= 10
        candidate_report["summary"]["by_instruction_id"][
            "keywords:existence"]["strict_passed"] -= 10
        candidate_report["summary"]["by_family"][
            "keywords"]["strict_passed"] -= 10
    arm_hashes = {"control": hashes(), "candidate": hashes()}
    control_status = status(
        "m1-122-control-fused-off", "0", arm_hashes["control"])
    candidate_status = status(
        "m1-122-candidate-fused-on", "1", arm_hashes["candidate"])
    if failed_gate:
        candidate_status["gates"]["ifeval"] = 1
    return M.compare(
        control_status=control_status,
        candidate_status=candidate_status,
        control_contract=contract("m1-122-control-fused-off", "0"),
        candidate_contract=contract("m1-122-candidate-fused-on", "1"),
        control_report=control_report,
        candidate_report=candidate_report,
        control_progress=progress(control_report, SHA),
        candidate_progress=progress(candidate_report, SHA),
        control_4xx=zero_4xx(),
        candidate_4xx=zero_4xx(),
        control_identity=identity(100, "1" * 32),
        candidate_identity=identity(200, "2" * 32),
        score_comparison=comparison(
            control_report, candidate_report, exact=False),
        exact_comparison=comparison(
            control_report, candidate_report, exact=True),
        paired_noninferiority=paired_comparison(
            control_report, candidate_report),
        file_sha256s=arm_hashes,
    )


class M1122IFEvalComparatorTest(unittest.TestCase):

    def test_exact_same_outputs_qualify(self) -> None:
        value = invoke()
        self.assertTrue(value["qualified"], value)
        self.assertTrue(value["strict_exact_output_qualified"])

    def test_output_drift_is_reported_but_score_non_regression_qualifies(
        self,
    ) -> None:
        value = invoke(output_drift=True)
        self.assertTrue(value["qualified"], value)
        self.assertFalse(value["strict_exact_output_qualified"])
        self.assertEqual(value["strict_exact_output_mismatch_count"], 64)

    def test_one_paired_regression_passes_five_point_screen(self) -> None:
        value = invoke(single_paired_regression=True)
        self.assertTrue(value["qualified"], value)
        self.assertFalse(value["strict_zero_stratum_qualified"])
        self.assertFalse(value["strict_exact_output_qualified"])
        self.assertEqual(value["strict_exact_output_mismatch_count"], 1)

    def test_statistically_clear_paired_regression_fails(self) -> None:
        value = invoke(paired_regression=True)
        self.assertFalse(value["qualified"])
        self.assertTrue(any(
            "paired non-inferiority screen did not qualify" in reason
            for reason in value["reasons"]
        ))

    def test_failed_lifecycle_gate_fails(self) -> None:
        value = invoke(failed_gate=True)
        self.assertFalse(value["qualified"])
        self.assertTrue(any(
            "service gates failed" in reason for reason in value["reasons"]))


class M1122IFEvalRunnerStaticTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.service = SERVICE_RUNNER.read_text(encoding="utf-8")
        cls.outer = OUTER_RUNNER.read_text(encoding="utf-8")
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")
        cls.api = IFEVAL_API.read_text(encoding="utf-8")

    def test_shell_syntax(self) -> None:
        subprocess.run(
            ["bash", "-n", str(SERVICE_RUNNER), str(OUTER_RUNNER),
             str(WRAPPER)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_ifeval_suite_is_offline_and_not_in_service_pythonpath(self) -> None:
        service_pythonpath = next(
            line for line in self.service.splitlines()
            if line.startswith("export PYTHONPATH=")
        )
        self.assertNotIn("IFEVAL_ENV", service_pythonpath)
        self.assertNotIn("google_ifeval", service_pythonpath)
        self.assertIn("IFEVAL_PYTHONPATH=", self.service)
        self.assertIn('NLTK_DATA="$IFEVAL_ENV/nltk_data"', self.service)
        self.assertIn("tests/ifeval_quality_api.py", self.service)
        self.assertIn("ifeval_install.json", self.service)
        self.assertIn("ifeval_environment.rc", self.service)
        self.assertIn("ifeval.checkpoint.json", self.service)
        self.assertIn("checkpoint_cleanup.rc", self.service)

    def test_current_policy_and_only_fused_selector_are_compared(self) -> None:
        self.assertIn("m1-122-control-fused-off", self.outer)
        self.assertIn("m1-122-candidate-fused-on", self.outer)
        self.assertIn("--allowed-switch fused_prefill", self.outer)
        self.assertIn("--require-exact-output", self.outer)
        self.assertIn(
            "compare_ifeval_paired_noninferiority.py", self.outer)
        self.assertIn("--paired-noninferiority", self.outer)
        self.assertIn("compare_m1_122_ifeval_service_ab.py", self.outer)
        self.assertIn("m1-122-fused-prefill-ifeval", self.wrapper)
        self.assertNotIn("computility-run.yaml", self.wrapper)

    def test_modern_lifecycle_and_hybrid64_are_required(self) -> None:
        for marker in (
            "exec_bi100_session.py",
            '"$ACTIVE_PGID" "$ACTIVE_PID" 60 20',
            "cleanup_recorded_bi100_sessions.py",
            "qualify_recorded_session_cleanup.py",
            "service_postflight_gate.py",
            "summarize_api_4xx_log.py",
        ):
            self.assertIn(marker, self.service)
        self.assertIn(
            'choices=("direct", "hybrid64", "aligned")', self.api)


if __name__ == "__main__":
    unittest.main()
