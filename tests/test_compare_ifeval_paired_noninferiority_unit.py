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
        spec = importlib.util.spec_from_file_location(
            "compare_ifeval_paired_noninferiority_unit", SCRIPT)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load IFEval paired comparator")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


M = load_module()
CONTRACT = json.loads(
    (ROOT / "quality/layered_quality_gate.v1.json").read_text(
        encoding="utf-8"))
MANIFEST = M.ifeval.canonical_manifest_contract(
    M.ifeval.EXPECTED_MANIFEST_SHA256)
assert MANIFEST is not None


def report(fused: bool) -> dict:
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
    counts = {"total": 64, "strict_passed": 64, "loose_passed": 64}
    return {
        "schema": "bi100-ifeval-result-v1",
        "version": 1,
        "qualified": True,
        "quality_run_eligible_for_baseline": True,
        "promotion_authorized": False,
        "manifest": {
            "sha256": M.ifeval.EXPECTED_MANIFEST_SHA256,
            "path_name": MANIFEST["path_name"],
            "subset_sha256": MANIFEST["subset_sha256"],
            "full_selection": True,
            "selected_keys": list(MANIFEST["selected_keys"]),
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
            "prompt_total": 64,
            "instruction_total": 64,
            "strict_prompt_passed": 64,
            "loose_prompt_passed": 64,
            "strict_instruction_passed": 64,
            "loose_instruction_passed": 64,
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
            for key in MANIFEST["selected_keys"]
        ],
        "privacy": {
            "contains_credentials": False,
            "contains_raw_prompts": False,
            "contains_raw_model_outputs": False,
            "contains_reasoning_text": False,
        },
    }


class CompareIFEvalPairedNoninferiorityTest(unittest.TestCase):

    def compare(self, baseline: dict, candidate: dict) -> dict:
        return M.compare(
            baseline,
            candidate,
            CONTRACT,
            allowed_switches={"fused_prefill"},
        )

    def test_64_clean_pairs_pass_five_point_screen_only(self) -> None:
        value = self.compare(report(False), report(True))
        self.assertEqual(value["status"], "pass")
        self.assertTrue(value["qualified"])
        self.assertFalse(value["promotion_power"]["sufficient"])
        self.assertEqual(
            value["promotion_power"]["minimum_zero_regression_samples"],
            149,
        )
        self.assertFalse(
            value["authorization"]["two_point_promotion_authorized"])

    def test_output_hash_drift_is_not_a_capability_failure(self) -> None:
        baseline = report(False)
        candidate = report(True)
        candidate["cases"][0]["semantic_output_sha256"] = "0" * 64
        value = self.compare(baseline, candidate)
        self.assertEqual(value["status"], "pass", value)

    def test_clear_paired_regression_fails(self) -> None:
        baseline = report(False)
        candidate = report(True)
        for case in candidate["cases"][:10]:
            case["strict"] = [False]
            case["semantic_output_sha256"] = "0" * 64
        candidate["summary"]["strict_prompt_passed"] -= 10
        candidate["summary"]["strict_instruction_passed"] -= 10
        candidate["summary"]["by_instruction_id"][
            "keywords:existence"]["strict_passed"] -= 10
        candidate["summary"]["by_family"][
            "keywords"]["strict_passed"] -= 10
        value = self.compare(baseline, candidate)
        self.assertEqual(value["status"], "fail", value)
        self.assertFalse(value["qualified"])

    def test_missing_case_outcome_is_invalid(self) -> None:
        baseline = report(False)
        candidate = report(True)
        del candidate["cases"][0]["strict"]
        value = self.compare(baseline, candidate)
        self.assertEqual(value["status"], "invalid", value)

    def test_report_retains_only_aggregate_outcomes(self) -> None:
        value = self.compare(report(False), report(True))
        encoded = json.dumps(value, sort_keys=True)
        self.assertNotIn('"baseline":', encoded)
        self.assertNotIn('"candidate":', encoded)
        self.assertTrue(all(
            item is False for item in value["privacy"].values()
        ))


if __name__ == "__main__":
    unittest.main()
