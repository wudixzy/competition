from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
SCRIPT = TESTS / "compare_ifeval_paired_noninferiority.py"


def load_module():
    sys.path.insert(0, str(TESTS))
    try:
        specification = importlib.util.spec_from_file_location(
            "compare_ifeval_power_noninferiority_unit", SCRIPT)
        if specification is None or specification.loader is None:
            raise RuntimeError("cannot load IFEval paired comparator")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


M = load_module()
CONTRACT = json.loads(
    (ROOT / "quality/layered_quality_gate.v2.json").read_text(
        encoding="utf-8"))


def report(fused: bool, count: int = 149) -> dict:
    manifest_sha = (
        M.ifeval.EXPECTED_POWER_MANIFEST_SHA256
        if count == 149 else M.ifeval.EXPECTED_MANIFEST_SHA256
    )
    manifest_contract = M.ifeval.canonical_manifest_contract(manifest_sha)
    if manifest_contract is None or manifest_contract["rows"] != count:
        raise ValueError("unsupported synthetic IFEval sample count")
    optimization = {
        "gdn_cache_policy": "admission64",
        "gdn_restore_mode": "hybrid64",
        "fused_prefill": fused,
        "kv_eviction_policy": "lru",
    }
    environment = {
        "BI100_GDN_CACHE_POLICY": "admission64",
        "BI100_GDN_RESTORE_MODE": "hybrid64",
        "BI100_ATTN_COREX_FUSED_PREFILL": "1" if fused else "0",
        "BI100_KV_EVICTION_POLICY": "lru",
        "FIXED": "same",
    }
    counts = {
        "total": count,
        "strict_passed": count,
        "loose_passed": count,
    }
    return {
        "schema": "bi100-ifeval-result-v1",
        "version": 1,
        "qualified": True,
        "quality_run_eligible_for_baseline": True,
        "promotion_authorized": False,
        "manifest": {
            "sha256": manifest_sha,
            "path_name": manifest_contract["path_name"],
            "subset_sha256": manifest_contract["subset_sha256"],
            "full_selection": True,
            "selected_keys": list(manifest_contract["selected_keys"]),
        },
        "runtime": {
            "source_revision": "a" * 40,
            "runtime_identity": "runtime",
            "runtime_overlay_sha256": "b" * 64,
            "runtime_contract_sha256": "c" * 64,
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
            "optimization": optimization,
        },
        "runtime_contract": {
            "sha256": "c" * 64,
            "file_sha256": "d" * 64,
            "contract": {
                "schema": "bi100-quality-runtime-contract-v1",
                "source_revision": "a" * 40,
                "command": ["fixed"],
                "environment": environment,
                "optimization_label": "candidate" if fused else "control",
            },
        },
        "request_conversion": {"temperature": 0, "seed": 20260725},
        "evaluator": {"revision": "e" * 40},
        "summary": {
            "prompt_total": count,
            "instruction_total": count,
            "strict_prompt_passed": count,
            "loose_prompt_passed": count,
            "strict_instruction_passed": count,
            "loose_instruction_passed": count,
            "by_instruction_id": {"keywords:existence": dict(counts)},
            "by_family": {"keywords": dict(counts)},
        },
        "cases": [
            {
                "key": key,
                "status": "pass",
                "instruction_id_list": ["keywords:existence"],
                "strict": [True],
                "loose": [True],
                "semantic_output_sha256": "f" * 64,
            }
            for key in manifest_contract["selected_keys"]
        ],
        "privacy": {
            "contains_credentials": False,
            "contains_raw_prompts": False,
            "contains_raw_model_outputs": False,
            "contains_reasoning_text": False,
        },
    }


def set_regressions(value: dict, count: int) -> None:
    for case in value["cases"][:count]:
        case["strict"] = [False]
        case["loose"] = [False]
        case["semantic_output_sha256"] = "0" * 64
    for name in (
        "strict_prompt_passed",
        "loose_prompt_passed",
        "strict_instruction_passed",
        "loose_instruction_passed",
    ):
        value["summary"][name] -= count
    for group, key in (
        ("by_instruction_id", "keywords:existence"),
        ("by_family", "keywords"),
    ):
        counts = value["summary"][group][key]
        counts["strict_passed"] -= count
        counts["loose_passed"] -= count


class IFEvalPowerNoninferiorityTest(unittest.TestCase):

    def compare(self, baseline: dict, candidate: dict) -> dict:
        return M.compare(
            baseline,
            candidate,
            CONTRACT,
            allowed_switches={"fused_prefill"},
        )

    def test_149_clean_pairs_authorize_only_two_point_surface(self) -> None:
        value = self.compare(report(False), report(True))
        self.assertEqual(value["schema"], M.SCHEMA_V2)
        self.assertEqual(value["version"], 2)
        self.assertEqual(value["status"], "pass", value)
        self.assertTrue(value["qualified"])
        self.assertEqual(value["screen"]["name"], "default-two-point")
        self.assertEqual(value["screen"]["noninferiority_margin"], 0.02)
        self.assertTrue(value["promotion_power"]["sufficient"])
        self.assertEqual(value["authorization"], {
            "five_point_screen_authorized": False,
            "two_point_capability_surface_authorized": True,
            "two_point_promotion_authorized": False,
            "overall_promotion_authorized": False,
        })

    def test_cross_arm_output_drift_is_not_a_task_failure(self) -> None:
        baseline = report(False)
        candidate = report(True)
        for case in candidate["cases"]:
            case["semantic_output_sha256"] = "0" * 64
        value = self.compare(baseline, candidate)
        self.assertEqual(value["status"], "pass", value)

    def test_clear_paired_regression_fails_two_point_gate(self) -> None:
        baseline = report(False)
        candidate = report(True)
        set_regressions(candidate, 10)
        value = self.compare(baseline, candidate)
        self.assertEqual(value["status"], "fail", value)
        self.assertFalse(value["qualified"])
        self.assertFalse(
            value["authorization"][
                "two_point_capability_surface_authorized"])

    def test_v2_contract_does_not_claim_two_point_power_for_64(self) -> None:
        value = self.compare(report(False, 64), report(True, 64))
        self.assertEqual(value["status"], "pass", value)
        self.assertEqual(
            value["screen"]["name"], "small-stratum-five-point")
        self.assertFalse(value["promotion_power"]["sufficient"])
        self.assertFalse(
            value["authorization"][
                "two_point_capability_surface_authorized"])

    def test_power_report_validation_rejects_wrong_count(self) -> None:
        baseline = report(False)
        candidate = copy.deepcopy(report(True))
        candidate["cases"].pop()
        value = self.compare(baseline, candidate)
        self.assertEqual(value["status"], "invalid", value)

    def test_power_report_validation_rejects_manifest_tamper(self) -> None:
        baseline = report(False)
        candidate = copy.deepcopy(report(True))
        candidate["manifest"]["selected_keys"][0:2] = reversed(
            candidate["manifest"]["selected_keys"][0:2])
        candidate["cases"][0:2] = reversed(candidate["cases"][0:2])
        value = self.compare(baseline, candidate)
        self.assertEqual(value["status"], "invalid", value)

        candidate = copy.deepcopy(report(True))
        candidate["manifest"]["subset_sha256"] = "0" * 64
        value = self.compare(baseline, candidate)
        self.assertEqual(value["status"], "invalid", value)


if __name__ == "__main__":
    unittest.main()
