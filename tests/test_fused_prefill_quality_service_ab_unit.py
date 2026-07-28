from __future__ import annotations

import copy
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
    "compare_fused_prefill_quality_service_ab",
    TESTS / "compare_fused_prefill_quality_service_ab.py",
)
COMPARE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(COMPARE)

import test_admission64_quality_service_ab_unit as fixtures


RUNNER = ROOT / "scripts/run_m1_85_admission64_quality_ab.sh"
WRAPPER = ROOT / "scripts/run_m1_112_fused_prefill_quality_ab.sh"


def contract(arm: str) -> dict:
    value = fixtures.contract("admission64")
    fused = "0" if arm == "control" else "1"
    value["environment"] = COMPARE.base.runtime_contract.service_environment(
        "/runtime/site-packages",
        gdn_cache_policy="admission64",
        gdn_restore_mode="hybrid64",
        fused_prefill=fused,
        kv_eviction_policy="lru",
        kernel_profile="submission",
    )
    value["optimization_label"] = (
        COMPARE.CONTROL_LABEL
        if arm == "control"
        else COMPARE.CANDIDATE_LABEL
    )
    return value


def status(arm: str) -> dict:
    fused = "0" if arm == "control" else "1"
    value = fixtures.status("admission64")
    value["label"] = (
        COMPARE.CONTROL_LABEL
        if arm == "control"
        else COMPARE.CANDIDATE_LABEL
    )
    value["optimization"] = COMPARE._optimization(fused)
    hashes = fixtures.FILE_SHA256S[arm]
    value["artifacts"] = {
        "runtime_contract_sha256": hashes["runtime_contract"],
        "quality_report_sha256": hashes["quality_report"],
        "agent_workload_sha256": hashes["agent_workload"],
        "api_4xx_attribution_sha256": hashes["api_4xx_attribution"],
        "process_group_identity_sha256": hashes[
            "process_group_identity"],
        "service_recovery_sha256": hashes["service_recovery"],
        "service_recovery_clean_sha256": hashes[
            "service_recovery_clean"],
    }
    return value


def compare_fixture(
    *,
    control_status: dict | None = None,
    candidate_status: dict | None = None,
    control_contract: dict | None = None,
    candidate_contract: dict | None = None,
    candidate_4xx: dict | None = None,
) -> dict:
    return COMPARE.compare(
        control_status=control_status or status("control"),
        candidate_status=candidate_status or status("candidate"),
        control_contract=control_contract or contract("control"),
        candidate_contract=candidate_contract or contract("candidate"),
        control_4xx=fixtures.api_4xx_report(),
        candidate_4xx=candidate_4xx or fixtures.api_4xx_report(),
        control_process_identity=fixtures.process_identity("fine32"),
        candidate_process_identity=fixtures.process_identity("admission64"),
        quality_comparison=fixtures.quality_comparison(),
        agent_comparison=fixtures.agent_comparison(),
        file_sha256s=copy.deepcopy(fixtures.FILE_SHA256S),
    )


class FusedPrefillQualityServiceAbTest(unittest.TestCase):

    def test_exact_fused_only_ab_qualifies(self):
        result = compare_fixture()
        self.assertTrue(result["qualified"], result)
        self.assertTrue(
            result["fused_prefill_quality_non_regression_authorized"])
        self.assertTrue(result["fused_only_runtime_delta_attested"])
        self.assertFalse(result["performance_authorized"])
        self.assertFalse(result["production_promotion_authorized"])

    def test_extra_environment_delta_is_rejected(self):
        candidate = contract("candidate")
        candidate["environment"]["BI100_UNDECLARED"] = "1"
        result = compare_fixture(candidate_contract=candidate)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "environment" in reason for reason in result["reasons"]))

    def test_wrong_candidate_restore_mode_is_rejected(self):
        candidate = status("candidate")
        candidate["optimization"]["gdn_restore_mode"] = "direct"
        result = compare_fixture(candidate_status=candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "candidate: optimization contract differs", result["reasons"])

    def test_failed_child_gate_is_rejected(self):
        candidate = status("candidate")
        candidate["gates"]["quality"] = 1
        result = compare_fixture(candidate_status=candidate)
        self.assertFalse(result["qualified"])
        self.assertTrue(any(
            "quality" in reason for reason in result["reasons"]))

    def test_4xx_shape_delta_is_rejected(self):
        candidate = fixtures.api_4xx_report()
        candidate["request_shapes"] = [{"endpoint": "chat"}]
        result = compare_fixture(candidate_4xx=candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "A/B 4xx attribution or request shapes differ", result["reasons"])

    def test_cli_binds_exact_arm_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm_hashes = {}
            for arm in ("control", "candidate"):
                arm_root = root / arm
                arm_root.mkdir()
                arm_hashes[arm] = {
                    "runtime_contract": fixtures.write_json(
                        arm_root / "runtime_contract.json", contract(arm)),
                    "quality_report": fixtures.write_json(
                        arm_root / "quality_report.json", {}),
                    "agent_workload": fixtures.write_json(
                        arm_root / "agent_workload.json", {}),
                    "api_4xx_attribution": fixtures.write_json(
                        arm_root / "api_4xx_attribution.json",
                        fixtures.api_4xx_report()),
                    "process_group_identity": fixtures.write_json(
                        arm_root / "process_group_identity.json",
                        fixtures.process_identity(
                            "fine32" if arm == "control"
                            else "admission64")),
                    "service_recovery": fixtures.write_json(
                        arm_root / "service_recovery.json", {}),
                    "service_recovery_clean": fixtures.write_json(
                        arm_root / "service_recovery_clean.json", {}),
                }
                arm_status = status(arm)
                arm_status["artifacts"] = {
                    f"{name}_sha256": digest
                    for name, digest in arm_hashes[arm].items()
                }
                fixtures.write_json(arm_root / "status.json", arm_status)

            quality = fixtures.quality_comparison()
            quality["inputs"] = {
                "baseline_file_sha256": arm_hashes["control"][
                    "quality_report"],
                "candidate_file_sha256": arm_hashes["candidate"][
                    "quality_report"],
            }
            agent = fixtures.agent_comparison()
            agent["inputs"] = {
                "baseline_file_sha256": arm_hashes["control"][
                    "agent_workload"],
                "candidate_file_sha256": arm_hashes["candidate"][
                    "agent_workload"],
            }
            quality_path = root / "quality-comparison.json"
            agent_path = root / "agent-comparison.json"
            output_path = root / "aggregate.json"
            fixtures.write_json(quality_path, quality)
            fixtures.write_json(agent_path, agent)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(TESTS / "compare_fused_prefill_quality_service_ab.py"),
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
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(report["qualified"], report)
            self.assertEqual(
                report["inputs"]["control_service_recovery_sha256"],
                arm_hashes["control"]["service_recovery"],
            )


class FusedPrefillQualityRunnerStaticTest(unittest.TestCase):

    def test_wrapper_selects_explicit_variant(self):
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn(
            "BI100_QUALITY_AB_VARIANT=m1-112-fused-prefill", source)
        self.assertIn(
            'exec "$ROOT/scripts/run_m1_85_admission64_quality_ab.sh" "$@"',
            source,
        )

    def test_runner_changes_only_fused_prefill(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("local restore_mode=direct", source)
        self.assertIn("local fused_prefill=0", source)
        self.assertIn("restore_mode=hybrid64", source)
        self.assertIn("fused_prefill=1", source)
        self.assertIn(
            "run_arm control admission64 m1-112-control-fused-off", source)
        self.assertIn(
            "run_arm candidate admission64 m1-112-candidate-fused-on",
            source,
        )
        self.assertIn(
            "compare_fused_prefill_quality_service_ab.py", source)
        self.assertNotIn("computility-run.yaml", source)

    def test_shell_syntax(self):
        for path in (RUNNER, WRAPPER):
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
