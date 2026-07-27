from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
SCRIPT = ROOT / "tests/compare_agent_workload_reports.py"
SPEC = importlib.util.spec_from_file_location("agent_compare", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def make_contract(policy: str) -> dict:
    runtime = MODULE.runtime_contract
    return {
        "schema": "bi100-quality-runtime-contract-v1",
        "version": 1,
        "source_revision": "a" * 40,
        "runtime_identity": "runtime-test",
        "runtime_overlay_sha256": "b" * 64,
        "instance": "private-instance",
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
        ),
        "cache_trace_enabled": True,
        "optimization_label": policy,
    }


def make_report(policy: str) -> dict:
    workload = MODULE.workload
    manifest, manifest_sha = workload.load_manifest(workload.DEFAULT_MANIFEST)
    contract = make_contract(policy)
    contract_sha = MODULE.runtime_contract.sha256_json(contract)
    cases = []
    for index, item in enumerate(manifest["cases"]):
        tool_case = (
            item["id"].startswith("forced_")
            or item["id"].startswith("auto_")
            or item["id"].startswith("stream_")
            or item["id"] == "large_tool_schema"
        )
        cases.append({
            "id": item["id"],
            "status": "pass",
            "error_type": "",
            "error_sha256": None,
            "observation": {
                "elapsed_s": 1.0,
                "finish_reason": "tool_calls" if tool_case else "stop",
                "content_chars": index,
                "reasoning_chars": 0,
                "tool_call_count": 1 if tool_case else 0,
                "prompt_tokens": 100 + index,
                "cached_tokens": 0,
                "completion_tokens": 8,
                "semantic_output_sha256": digest(item["id"]),
                "facts": {"rule_passed": True},
            },
        })
    return {
        "schema": workload.REPORT_SCHEMA,
        "version": workload.REPORT_VERSION,
        "qualified": True,
        "promotion_authorized": False,
        "label": policy,
        "created_at_utc": "2026-07-24T00:00:00Z",
        "run_id_sha256": digest(policy),
        "manifest": {
            "path_name": workload.DEFAULT_MANIFEST.name,
            "sha256": manifest_sha,
            "revision": manifest["revision"],
            "case_count": len(manifest["cases"]),
        },
        "runtime": {
            "source_revision": contract["source_revision"],
            "runtime_identity": contract["runtime_identity"],
            "runtime_overlay_sha256": contract["runtime_overlay_sha256"],
            "runtime_contract_sha256": contract_sha,
            "instance": contract["instance"],
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "max_model_len": 262144,
        },
        "runtime_contract": {
            "sha256": contract_sha,
            "contract": contract,
        },
        "generator": {
            "runner_sha256": "c" * 64,
            "seed": manifest["seed"],
        },
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_tool_arguments": False,
            "contains_credentials": False,
        },
        "summary": {
            "complete": True,
            "passed": len(manifest["cases"]),
            "failed": 0,
            "total": len(manifest["cases"]),
        },
        "cases": cases,
    }


class AgentWorkloadComparisonUnitTest(unittest.TestCase):

    def test_exact_quality_with_allowed_policy_change_passes(self):
        report = MODULE.compare_reports(
            make_report("fine32"), make_report("admission64"))
        self.assertTrue(report["qualified"])
        self.assertTrue(report["agent_quality_non_regression_authorized"])
        self.assertFalse(report["overall_promotion_authorized"])

    def test_semantic_output_change_fails(self):
        baseline = make_report("fine32")
        candidate = make_report("admission64")
        candidate["cases"][0]["observation"][
            "semantic_output_sha256"] = "d" * 64
        report = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "forced_terminal: semantic_output_sha256 differs",
            report["reasons"],
        )

    def test_disallowed_environment_change_fails(self):
        baseline = make_report("fine32")
        candidate = make_report("admission64")
        candidate = copy.deepcopy(candidate)
        contract = candidate["runtime_contract"]["contract"]
        contract["environment"]["BI100_UNDECLARED"] = "1"
        contract_sha = MODULE.runtime_contract.sha256_json(contract)
        candidate["runtime_contract"]["sha256"] = contract_sha
        candidate["runtime"]["runtime_contract_sha256"] = contract_sha
        report = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(report["qualified"])
        self.assertTrue(any(
            "disallowed runtime environment" in reason
            for reason in report["reasons"]))

    def test_combined_qk_is_the_only_allowed_kernel_profile_delta(self):
        baseline = make_report("fine32")
        candidate = make_report("fine32")
        for report, profile in (
                (baseline, "strict-reference"),
                (candidate, "strict-reference-combined-qk")):
            contract = report["runtime_contract"]["contract"]
            contract["environment"] = (
                MODULE.runtime_contract.service_environment(
                    "/runtime/site-packages",
                    gdn_cache_policy="fine32",
                    gdn_restore_mode="direct",
                    fused_prefill="0",
                    kv_eviction_policy="lru",
                    kernel_profile=profile,
                ))
            contract_sha = MODULE.runtime_contract.sha256_json(contract)
            report["runtime_contract"]["sha256"] = contract_sha
            report["runtime"]["runtime_contract_sha256"] = contract_sha
        self.assertTrue(
            MODULE.compare_reports(baseline, candidate)["qualified"])

        contract = candidate["runtime_contract"]["contract"]
        contract["environment"]["BI100_MOE_COREX_DIRECT_ROUTED"] = "1"
        contract_sha = MODULE.runtime_contract.sha256_json(contract)
        candidate["runtime_contract"]["sha256"] = contract_sha
        candidate["runtime"]["runtime_contract_sha256"] = contract_sha
        self.assertFalse(
            MODULE.compare_reports(baseline, candidate)["qualified"])

    def test_privacy_declaration_fails_closed(self):
        baseline = make_report("fine32")
        candidate = make_report("admission64")
        candidate["privacy"]["contains_raw_model_outputs"] = True
        report = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "candidate: privacy declaration is invalid", report["reasons"])


if __name__ == "__main__":
    unittest.main()
