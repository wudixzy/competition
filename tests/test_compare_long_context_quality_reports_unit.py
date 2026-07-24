from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
SCRIPT = ROOT / "tests/compare_long_context_quality_reports.py"
SPEC = importlib.util.spec_from_file_location("long_compare", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def digest(character: str) -> str:
    return character * 64


def repeated_pairs(case_id: str) -> tuple[tuple[int, int], ...]:
    return {
        "4k_cold_warm_recall": ((0, 1),),
        "32k_partial_branch": ((0, 1),),
        "32k_multimodal_isolation": ((0, 1),),
        "65k_multiturn_large_tools": ((0, 1),),
        "65k_long_tool_result": ((0, 1),),
        "65k_interleaved_sessions": ((0, 2), (1, 3)),
        "131k_cold_warm_recall": ((0, 1),),
        "235k_agent_large_output_budget": ((0, 1),),
        "235k_partial_branch": ((0, 1), (2, 3)),
        "near_262k_capacity": ((0, 1),),
    }.get(case_id, ())


def cached_pattern(case_id: str, target: int) -> list[int]:
    return {
        "short_basic_recall": [0],
        "4k_cold_warm_recall": [0, target - 16],
        "32k_partial_branch": [0, target - 16, target - 32],
        "32k_multimodal_isolation": [0, target - 16, 0],
        "65k_multiturn_large_tools": [0, target - 16],
        "65k_long_tool_result": [0, target - 16],
        "65k_interleaved_sessions": [0, 0, target - 16, target - 16],
        "131k_cold_warm_recall": [0, target - 16],
        "131k_reasoning_recall": [0],
        "235k_agent_large_output_budget": [0, target - 16],
        "235k_partial_branch": [0, target - 16, target - 32, target - 16],
        "near_262k_capacity": [0, target - 16, 0],
    }[case_id]


def construction(case: dict, target: int, index: int) -> dict:
    case_id = case["id"]
    tool_sha = digest("a")
    if case_id == "65k_multiturn_large_tools":
        tool_sha = MODULE.manifest_validator.EXPECTED_GENERATED_ASSETS[
            "large_tools_65k_sha256"]
    elif case_id == "65k_long_tool_result":
        tool_sha = MODULE.manifest_validator.EXPECTED_GENERATED_ASSETS[
            "fetch_record_tool_sha256"]
    elif case_id == "235k_agent_large_output_budget":
        tool_sha = MODULE.manifest_validator.EXPECTED_GENERATED_ASSETS[
            "large_tools_235k_sha256"]
    return {
        "schema": "bi100-exact-chat-prompt-v1",
        "target_prompt_tokens": target,
        "local_prompt_tokens": target,
        "fixed_prompt_tokens": 64,
        "filler_token_ids_requested": target - 64,
        "filler_text_sha256": digest("1"),
        "filler_source_sha256": digest("2"),
        "rendered_prompt_token_ids_sha256": digest("3"),
        "messages_sha256": digest(str((index + 4) % 10)),
        "tools_sha256": tool_sha,
        "thinking": case_id in {
            "131k_reasoning_recall", "235k_agent_large_output_budget"},
        "template_kwargs_mode": "direct",
        "attempts": 2,
    }


def facts(case_id: str) -> dict:
    value = {name: True for name in MODULE.TRUE_FACTS[case_id]}
    if case_id in {
            "65k_multiturn_large_tools",
            "235k_agent_large_output_budget"}:
        value["tool_count"] = 92
    if case_id == "32k_multimodal_isolation":
        assets = MODULE.manifest_validator.EXPECTED_GENERATED_ASSETS
        value["red_image_sha256"] = assets["red_png_data_url_sha256"]
        value["blue_image_sha256"] = assets["blue_png_data_url_sha256"]
        value["cache_trace_records_sha256"] = digest("c")
        value["cache_trace_version"] = 4
    return value


def case_result(case: dict) -> dict:
    case_id = case["id"]
    count = MODULE.REQUEST_COUNTS[case_id]
    targets = [case["target_prompt_tokens"]] * count
    if case_id == "near_262k_capacity":
        targets[-1] -= 1
    cached = cached_pattern(case_id, case["target_prompt_tokens"])
    requests = []
    for index, (target, hit) in enumerate(zip(targets, cached)):
        prompt = target + (64 if case_id == "32k_multimodal_isolation" else 0)
        completion = max(case["min_completion_tokens"], 2)
        requests.append({
            "status": 200,
            "model": "llm",
            "local_prompt_tokens": target,
            "prompt_tokens": prompt,
            "cached_tokens": hit,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "finish_reason": (
                "tool_calls" if case_id in {
                    "65k_multiturn_large_tools",
                    "235k_agent_large_output_budget"} else "stop"),
            "semantic_output_sha256": digest(str((index + 1) % 10)),
            "content_sha256": digest(str((index + 2) % 10)),
            "reasoning_sha256": digest(str((index + 3) % 10)),
            "tool_calls_sha256": digest(str((index + 4) % 10)),
            "first_generated_token_sha256": (
                digest("f") if case_id in MODULE.NEXT_TOKEN_IDS else None),
            "request_contract_sha256": digest(str((index + 5) % 10)),
            "token_accounting": (
                "local_template_plus_vision"
                if case_id == "32k_multimodal_isolation"
                else "server_exact"),
            "protocol_validated": True,
            "elapsed_s": 1.0,
        })
    for left, right in repeated_pairs(case_id):
        for field in (
                "request_contract_sha256", "semantic_output_sha256",
                "completion_tokens", "finish_reason", "content_sha256",
                "reasoning_sha256", "tool_calls_sha256"):
            requests[right][field] = requests[left][field]
    construction_targets = [case["target_prompt_tokens"]] * (
        MODULE.CONSTRUCTION_COUNTS[case_id])
    if case_id == "near_262k_capacity":
        construction_targets[-1] -= 1
    observation = {
        "requests": requests,
        "construction": [
            construction(case, target, index)
            for index, target in enumerate(construction_targets)
        ],
        "facts": facts(case_id),
    }
    return {
        **case,
        "status": "pass",
        "ok": True,
        "elapsed_s": 2.0,
        "error_code": "",
        "observation": observation,
    }


def valid_report() -> dict:
    manifest, manifest_sha = MODULE._load_manifest(MODULE.DEFAULT_MANIFEST)
    cases = [case_result(case) for case in manifest["cases"]]
    files = [{"name": "tokenizer.json", "bytes": 10,
              "sha256": digest("a")}]
    command = MODULE.runtime_contract.service_command("/model")
    environment = MODULE.runtime_contract.service_environment(
        "/runtime/site-packages",
        gdn_cache_policy="fine32",
        gdn_restore_mode="direct",
        fused_prefill="0",
        kv_eviction_policy="lru",
    )
    contract = {
        "schema": "bi100-quality-runtime-contract-v1",
        "version": 1,
        "source_revision": "a" * 40,
        "runtime_identity": "corex-unit",
        "runtime_overlay_sha256": digest("1"),
        "instance": "unit-tp4",
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": "/model",
        "tokenizer_path": "/model",
        "served_model_name": "llm",
        "base_image": MODULE.BASE_IMAGE,
        "command": command,
        "environment": environment,
        "cache_trace_enabled": True,
        "optimization_label": "baseline",
    }
    return {
        "schema": MODULE.REPORT_SCHEMA,
        "version": 2,
        "qualified": True,
        "quality_run_eligible_for_baseline": True,
        "overall_promotion_authorized": False,
        "label": "unit",
        "run_id_sha256": digest("0"),
        "created_at_utc": "2026-07-24T00:00:00+00:00",
        "manifest": {
            "path_name": MODULE.DEFAULT_MANIFEST.name,
            "sha256": manifest_sha,
            "total_cases": 12,
            "seed": 20260724,
        },
        "runtime": {
            "source_revision": "a" * 40,
            "runtime_identity": "corex-unit",
            "runtime_overlay_sha256": digest("1"),
            "service_command_sha256": MODULE._sha256_json(command),
            "service_env_sha256": MODULE._sha256_json(environment),
            "instance": "unit-tp4",
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "model_path": "/model",
            "max_model_len": 262144,
            "served_model_name": "llm",
            "fresh_service_attested": True,
            "cache_trace_v4_attested": True,
            "model_list_contract_sha256": digest("4"),
        },
        "runtime_contract": {
            "sha256": MODULE._sha256_json(contract),
            "contract": contract,
        },
        "generator": {
            "runner_sha256": digest("5"),
            "exact_prompt_module_sha256": digest("6"),
            "transformers_version": "unit",
        },
        "tokenizer": {
            "tokenizer_class": "UnitTokenizer",
            "artifact_set_sha256": MODULE._sha256_json(files),
            "chat_template_sha256": digest("7"),
            "files": files,
            "template_kwargs_mode": "direct",
            "thinking_false_prompt_sha256": digest("8"),
            "thinking_true_prompt_sha256": digest("9"),
            "thinking_modes_distinct": True,
        },
        "selection": {
            "tier": "extended", "explicit_cases": [], "selected_cases": 12,
        },
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_credentials": False,
        },
        "summary": {
            "passed": 12, "failed": 0, "total": 12,
            "selected_total": 12, "complete": True,
            "pass_rate": 1.0, "wall_s": 24.0,
        },
        "cases": cases,
    }


def refresh_runtime_contract(report: dict) -> None:
    wrapper = report["runtime_contract"]
    contract = wrapper["contract"]
    wrapper["sha256"] = MODULE._sha256_json(contract)
    report["runtime"]["service_command_sha256"] = MODULE._sha256_json(
        contract["command"])
    report["runtime"]["service_env_sha256"] = MODULE._sha256_json(
        contract["environment"])


class LongContextQualityComparisonTest(unittest.TestCase):

    def compare(self, baseline: dict, candidate: dict) -> dict:
        return MODULE.compare_reports(baseline, candidate)

    def test_identical_complete_reports_qualify_without_overall_promotion(self):
        report = valid_report()
        result = self.compare(report, copy.deepcopy(report))
        self.assertTrue(result["qualified"])
        self.assertTrue(result["long_context_quality_non_regression_authorized"])
        self.assertFalse(result["overall_promotion_authorized"])

    def test_exact_output_regression_fails(self):
        baseline = valid_report()
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["observation"]["requests"][0][
            "semantic_output_sha256"] = digest("e")
        result = self.compare(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertTrue(any("semantic_output_sha256 differs" in reason
                            for reason in result["reasons"]))

    def test_next_token_regression_fails(self):
        baseline = valid_report()
        candidate = copy.deepcopy(baseline)
        candidate["cases"][7]["observation"]["requests"][0][
            "first_generated_token_sha256"] = digest("e")
        candidate["cases"][7]["observation"]["requests"][1][
            "first_generated_token_sha256"] = digest("e")
        self.assertFalse(self.compare(baseline, candidate)["qualified"])

    def test_semantic_output_may_differ_when_independent_rules_hold(self):
        baseline = valid_report()
        candidate = copy.deepcopy(baseline)
        candidate["cases"][8]["observation"]["requests"][0][
            "semantic_output_sha256"] = digest("e")
        self.assertTrue(self.compare(baseline, candidate)["qualified"])

    def test_request_or_tokenizer_contract_drift_fails(self):
        baseline = valid_report()
        candidate = copy.deepcopy(baseline)
        candidate["cases"][8]["observation"]["requests"][0][
            "request_contract_sha256"] = digest("e")
        candidate["tokenizer"]["chat_template_sha256"] = digest("e")
        result = self.compare(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "baseline and candidate tokenizer identities differ",
            result["reasons"],
        )

    def test_multimodal_cross_image_hit_fails(self):
        baseline = valid_report()
        candidate = copy.deepcopy(baseline)
        candidate["cases"][3]["observation"]["requests"][2][
            "cached_tokens"] = 16
        self.assertFalse(self.compare(baseline, candidate)["qualified"])

    def test_only_declared_environment_may_differ_between_ab_runs(self):
        baseline = valid_report()
        candidate = copy.deepcopy(baseline)
        candidate["runtime_contract"]["contract"]["environment"][
            "BI100_GDN_CACHE_POLICY"] = "admission64"
        candidate["runtime_contract"]["contract"][
            "optimization_label"] = "candidate"
        refresh_runtime_contract(candidate)
        self.assertTrue(self.compare(baseline, candidate)["qualified"])

        candidate = copy.deepcopy(baseline)
        candidate["runtime_contract"]["contract"]["environment"][
            "BI100_UNDOCUMENTED_FAST_PATH"] = "1"
        refresh_runtime_contract(candidate)
        self.assertFalse(self.compare(baseline, candidate)["qualified"])

    def test_ab_requires_same_source_overlay_and_instance(self):
        baseline = valid_report()
        for field, value in (
                ("source_revision", "1" * 40),
                ("runtime_identity", "different-runtime"),
                ("runtime_overlay_sha256", "2" * 64),
                ("instance", "different-instance")):
            with self.subTest(field=field):
                candidate = copy.deepcopy(baseline)
                candidate["runtime"][field] = value
                candidate["runtime_contract"]["contract"][field] = value
                refresh_runtime_contract(candidate)
                self.assertFalse(self.compare(
                    baseline, candidate)["qualified"])

    def test_raw_or_incomplete_report_fails_without_crashing(self):
        baseline = valid_report()
        candidate = copy.deepcopy(baseline)
        candidate["privacy"]["contains_raw_model_outputs"] = True
        candidate["cases"][-1]["observation"] = {}
        result = self.compare(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertTrue(result["reasons"])


if __name__ == "__main__":
    unittest.main()
