from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def load(name: str):
    path = TESTS / f"{name}.py"
    sys.path.insert(0, str(TESTS))
    try:
        specification = importlib.util.spec_from_file_location(name, path)
        if specification is None or specification.loader is None:
            raise RuntimeError(f"cannot load {name}")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


M = load("compare_m1_137_ifeval_power_ab")
POWER = load("test_ifeval_power_noninferiority_unit")


def paired_report() -> dict:
    baseline = POWER.report(False)
    candidate = POWER.report(True)
    value = POWER.M.compare(
        baseline,
        candidate,
        POWER.CONTRACT,
        allowed_switches={"fused_prefill"},
    )
    value.update({
        "baseline_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "contract_sha256": M.LAYERED_CONTRACT_SHA256,
    })
    return value


class M1137IFEvalPowerABTest(unittest.TestCase):

    def test_paired_power149_contract_qualifies_only_capability_surface(self):
        value = paired_report()
        self.assertEqual(
            M._paired_reasons(
                value,
                baseline_sha256="a" * 64,
                candidate_sha256="b" * 64,
            ),
            [],
        )
        self.assertFalse(
            value["authorization"]["overall_promotion_authorized"])
        self.assertTrue(
            value["authorization"][
                "two_point_capability_surface_authorized"])

    def test_paired_authorization_drift_is_rejected(self):
        value = paired_report()
        value["authorization"]["overall_promotion_authorized"] = True
        self.assertIn(
            "power149 authorization boundary differs",
            M._paired_reasons(
                value,
                baseline_sha256="a" * 64,
                candidate_sha256="b" * 64,
            ),
        )

    def test_historical_layered_contract_digest_is_pinned_separately(self):
        self.assertNotEqual(
            M.service._file_sha256(M.LAYERED_CONTRACT),
            M.EXPECTED_LAYERED_CONTRACT_SHA256,
        )
        self.assertEqual(
            M.CURRENT_LAYERED_CONTRACT_SHA256,
            M.service._file_sha256(M.LAYERED_CONTRACT),
        )
        self.assertEqual(
            M.LAYERED_CONTRACT_SHA256,
            M.EXPECTED_LAYERED_CONTRACT_SHA256,
        )

    def test_progress_and_install_are_bound_to_149_manifest(self):
        report = {"run_id_sha256": "c" * 64}
        progress = {
            "schema": "bi100-ifeval-progress-v1",
            "version": 1,
            "run_id_sha256": "c" * 64,
            "selected": 149,
            "attempted": 149,
            "successful": 149,
            "errors": 0,
            "last_ordinal": 149,
            "complete": True,
            "report_sha256": "d" * 64,
            "failures": [],
            "privacy": {
                "contains_credentials": False,
                "contains_raw_prompts": False,
                "contains_raw_model_outputs": False,
                "contains_reasoning_text": False,
            },
        }
        self.assertEqual(
            M._progress_reasons(
                progress, report, "d" * 64, "control"),
            [],
        )
        install = {
            "schema": "bi100-ifeval-offline-environment-v1",
            "version": 1,
            "qualified": True,
            "manifest_sha256": M.MANIFEST_SHA256,
            "python": "3.10.12",
            "system_site_packages_modified": False,
            "punkt_tab_archive_sha256": M.EXPECTED_PUNKT_TAB_SHA256,
            "distribution_sha256": dict(
                M.EXPECTED_DISTRIBUTION_SHA256),
        }
        self.assertEqual(M._install_reasons(install, "control"), [])
        changed = copy.deepcopy(install)
        changed["manifest_sha256"] = "0" * 64
        self.assertTrue(M._install_reasons(changed, "control"))
        changed = copy.deepcopy(install)
        changed["distribution_sha256"].pop(
            next(iter(changed["distribution_sha256"])))
        self.assertTrue(M._install_reasons(changed, "control"))
        changed = copy.deepcopy(install)
        changed["punkt_tab_archive_sha256"] = "0" * 64
        self.assertTrue(M._install_reasons(changed, "control"))

    def test_runner_is_private_experiment_only(self):
        wrapper = (
            ROOT / "scripts/run_m1_137_ifeval_power_ab.sh"
        ).read_text(encoding="utf-8")
        orchestrator = (
            ROOT / "scripts/run_m1_85_admission64_quality_ab.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "m1-137-fused-prefill-ifeval-power149", wrapper)
        self.assertIn("manifest.power149.v2.json", wrapper)
        self.assertIn(
            "compare_m1_137_ifeval_power_ab.py", orchestrator)
        self.assertIn("layered_quality_gate.v2.json", orchestrator)
        service_runner = (
            ROOT / "scripts/run_quality_service_gate.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'value.get("distribution_sha256") != expected_distributions',
            service_runner,
        )
        self.assertIn(
            'value.get("punkt_tab_archive_sha256")',
            service_runner,
        )
        self.assertNotIn("computility-run.yaml", wrapper)
        self.assertNotIn("git push", wrapper)

    def test_legacy_m1_122_loader_includes_declared_install_artifact(self):
        paths = M.legacy._arm_paths(Path("/tmp/unit"))
        self.assertIn("ifeval_install", paths)


if __name__ == "__main__":
    unittest.main()
