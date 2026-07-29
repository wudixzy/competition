from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/compare_ifeval_reports.py"
SPEC = importlib.util.spec_from_file_location("compare_ifeval_reports", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def report(policy: str) -> dict:
    optimization = {
        "gdn_cache_policy": policy,
        "gdn_restore_mode": "direct",
        "fused_prefill": False,
        "kv_eviction_policy": "lru",
    }
    environment = {
        "BI100_GDN_CACHE_POLICY": policy,
        "BI100_GDN_RESTORE_MODE": "direct",
        "BI100_ATTN_COREX_FUSED_PREFILL": "0",
        "BI100_KV_EVICTION_POLICY": "lru",
        "FIXED": "same",
    }
    counts = {"total": 64, "strict_passed": 50, "loose_passed": 55}
    return {
        "schema": "bi100-ifeval-result-v1",
        "version": 1,
        "qualified": True,
        "quality_run_eligible_for_baseline": True,
        "promotion_authorized": False,
        "manifest": {
            "sha256": MODULE.EXPECTED_MANIFEST_SHA256,
            "full_selection": True,
            "selected_keys": list(range(64)),
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
                "optimization_label": policy,
            },
        },
        "request_conversion": {"temperature": 0, "seed": 20260725},
        "evaluator": {"revision": "e" * 40},
        "summary": {
            "prompt_total": 64,
            "instruction_total": 64,
            "strict_prompt_passed": 50,
            "loose_prompt_passed": 55,
            "strict_instruction_passed": 50,
            "loose_instruction_passed": 55,
            "by_instruction_id": {"keywords:existence": dict(counts)},
            "by_family": {"keywords": dict(counts)},
        },
        "cases": [
            {
                "key": key,
                "status": "pass",
                "instruction_id_list": ["keywords:existence"],
                "strict": [key < 50],
                "loose": [key < 55],
                "semantic_output_sha256": "f" * 64,
            }
            for key in range(64)
        ],
        "privacy": {
            "contains_credentials": False,
            "contains_raw_prompts": False,
            "contains_raw_model_outputs": False,
            "contains_reasoning_text": False,
        },
    }


class CompareIFEvalReportsTest(unittest.TestCase):

    def test_declared_cache_policy_difference_qualifies(self):
        reasons = MODULE.comparison_reasons(
            report("fine32"), report("admission64"),
            {"gdn_cache_policy"}, True)
        self.assertEqual(reasons, [])

    def test_aggregate_and_per_instruction_regression_fail(self):
        baseline = report("fine32")
        candidate = report("admission64")
        candidate["summary"]["strict_prompt_passed"] -= 1
        candidate["summary"]["strict_instruction_passed"] -= 1
        candidate["summary"]["by_instruction_id"][
            "keywords:existence"]["strict_passed"] -= 1
        candidate["summary"]["by_family"][
            "keywords"]["strict_passed"] -= 1
        candidate["cases"][0]["strict"] = [False]
        reasons = MODULE.comparison_reasons(
            baseline, candidate, {"gdn_cache_policy"}, False)
        self.assertIn(
            "candidate regressed aggregate strict_prompt_passed", reasons)
        self.assertIn(
            "candidate regressed by_instruction_id keywords:existence "
            "strict_passed", reasons)

    def test_summary_case_mismatch_is_invalid(self):
        baseline = report("fine32")
        candidate = report("admission64")
        candidate["summary"]["strict_prompt_passed"] -= 1
        reasons = MODULE.comparison_reasons(
            baseline, candidate, {"gdn_cache_policy"}, False)
        self.assertIn(
            "candidate: summary differs from cases in "
            "strict_prompt_passed",
            reasons,
        )

    def test_undeclared_environment_difference_fails(self):
        baseline = report("fine32")
        candidate = report("admission64")
        candidate["runtime_contract"]["contract"]["environment"][
            "FIXED"] = "changed"
        reasons = MODULE.comparison_reasons(
            baseline, candidate, {"gdn_cache_policy"}, False)
        self.assertIn(
            "candidate runtime environment differs: FIXED", reasons)

    def test_exact_output_gate_detects_drift(self):
        baseline = report("fine32")
        candidate = report("admission64")
        candidate["cases"][3]["semantic_output_sha256"] = "0" * 64
        reasons = MODULE.comparison_reasons(
            baseline, candidate, {"gdn_cache_policy"}, True)
        self.assertIn("candidate output differs for key 3", reasons)


if __name__ == "__main__":
    unittest.main()
