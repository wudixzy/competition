from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
MANIFEST = json.loads((
    ROOT / "quality/official_metrics_manifest.v1.json"
).read_text(encoding="utf-8"))
MANIFEST_SHA = hashlib.sha256((
    ROOT / "quality/official_metrics_manifest.v1.json"
).read_bytes()).hexdigest()
SCRIPT = ROOT / "tests/compare_quality_gate_reports.py"
SPEC = importlib.util.spec_from_file_location("quality_compare", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_report(label: str) -> dict:
    cases = []
    groups = {}
    for case in MANIFEST["cases"]:
        case_id = case["id"]
        status_codes = [200]
        finish_reasons = ["stop"]
        prompt_tokens = [10]
        cached_tokens = [0]
        completion_tokens = [2]
        facts = {
            key: True for key in MODULE.TRUE_FACTS.get(case_id, ())
        }
        if case_id in MODULE.ALWAYS_REJECTED:
            status_codes = [400]
            finish_reasons = []
            prompt_tokens = []
            cached_tokens = []
            completion_tokens = []
        if case_id in MODULE.PARAMETER_FACTS:
            parameter, accepted = MODULE.PARAMETER_FACTS[case_id]
            facts["parameter"] = parameter
            facts["accepted"] = (
                status_codes == [200] if accepted is None else accepted)
        if case_id in MODULE.MAX_TOKEN_FACTS:
            facts["requested_max_tokens"] = MODULE.MAX_TOKEN_FACTS[case_id]
        if case_id == "max_tokens_1":
            finish_reasons = ["length"]
            completion_tokens = [1]
        if case_id in ("tool_calling", "function_calling"):
            facts["tool_calls"] = 1
            finish_reasons = ["tool_calls"]
        if case_id == "reasoning":
            facts["reasoning_present"] = True
        if case_id in ("multimodal_input", "base64_png"):
            status_codes = [200, 200, 200]
            finish_reasons = ["stop", "stop", "stop"]
            prompt_tokens = [64, 64, 64]
            cached_tokens = [0, 48, 0]
            completion_tokens = [8, 8, 8]
        if case_id == "prefix_cache_hit":
            status_codes = [200, 200]
            finish_reasons = ["stop", "stop"]
            prompt_tokens = [100, 100]
            cached_tokens = [0, 80]
            completion_tokens = [2, 2]
        if case_id == "idempotency":
            status_codes = [200, 200]
            finish_reasons = ["stop", "stop"]
            prompt_tokens = [10, 10]
            cached_tokens = [0, 0]
            completion_tokens = [2, 2]
        thinking = {
            "thinking_disabled_top_level": (False, False),
            "thinking_true": (True, True),
            "thinking_false": (False, False),
            "thinking_default": (True, True),
        }
        if case_id in thinking:
            enabled, reasoning = thinking[case_id]
            facts.update({
                "thinking_enabled": enabled,
                "reasoning_present": reasoning,
            })
        if case_id == "thinking_disabled_top_level":
            facts["request_protocol"] = "top_level"
        if case_id == "n_1":
            facts["n"] = 1
        if case_id == "n_2":
            facts["n"] = 2
            finish_reasons = ["stop", "stop"]
        if case_id in ("streaming_usage", "streaming_sse_usage"):
            facts.update({"chunks": 12, "done": 1, "usage_blocks": 1})
        if case_id == "exact_output_truncation":
            finish_reasons = ["length"]
            completion_tokens = [32768]
            facts["exact_completion_tokens"] = 32768
        cases.append({
            **case,
            "ok": True,
            "status": "pass",
            "skip_reason": "",
            "elapsed_s": 1.0,
            "error_code": "",
            "observation": {
                "status_codes": status_codes,
                "finish_reasons": finish_reasons,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
                "semantic_output_sha256": "a" * 64,
                "facts": facts,
            },
        })
        groups.setdefault(case["group"], {
            "passed": 0,
            "skipped": 0,
            "failed": 0,
            "total": 0,
            "pass_rate": 1.0,
        })
        groups[case["group"]]["passed"] += 1
        groups[case["group"]]["total"] += 1
    revision = "a" * 40
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
        "source_revision": revision,
        "runtime_identity": "corex-unit-runtime",
        "runtime_overlay_sha256": "b" * 64,
        "instance": "private-instance",
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": "/model",
        "tokenizer_path": "/model",
        "served_model_name": "llm",
        "base_image": MODULE.runtime_contract.BASE_IMAGE,
        "command": command,
        "environment": environment,
        "cache_trace_enabled": True,
        "optimization_label": label,
    }
    files = [{"name": "tokenizer.json", "bytes": 10,
              "sha256": "c" * 64}]
    return {
        "schema": "bi100-quality-gate-result-v1",
        "version": 1,
        "qualified": True,
        "quality_run_eligible_for_baseline": True,
        "promotion_authorized": False,
        "label": label,
        "run_id_sha256": "d" * 64,
        "created_at_utc": "2026-07-24T00:00:00+00:00",
        "manifest": {
            "path_name": "official_metrics_manifest.v1.json",
            "sha256": MANIFEST_SHA,
            "source_sha256": MANIFEST["source"]["sha256"],
            "total_cases": 53,
        },
        "runtime": {
            "source_revision": revision,
            "runtime_identity": "corex-unit-runtime",
            "runtime_overlay_sha256": "b" * 64,
            "service_command_sha256": MODULE.runtime_contract.sha256_json(
                command),
            "service_env_sha256": MODULE.runtime_contract.sha256_json(
                environment),
            "instance": "private-instance",
            "gpu_count": 4,
            "tensor_parallel_size": 4,
            "model_path": "/model",
            "tokenizer_path": "/model",
            "max_model_len": 262144,
            "model": "llm",
            "endpoint_mode": "direct",
            "allow_bare_engine_n2_skip": False,
            "cache_trace_v4_attested": True,
        },
        "runtime_contract": {
            "sha256": MODULE.runtime_contract.sha256_json(contract),
            "contract": contract,
        },
        "generator": {
            "runner_sha256": "e" * 64,
            "transformers_version": "unit",
        },
        "tokenizer": {
            "tokenizer_class": "UnitTokenizer",
            "artifact_set_sha256": MODULE.runtime_contract.sha256_json(files),
            "chat_template_sha256": "f" * 64,
            "files": files,
        },
        "selection": {
            "tier": "extended",
            "explicit_cases": [],
            "selected_cases": 53,
            "allowed_skip_ids": [],
        },
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_credentials": False,
        },
        "summary": {
            "complete": True,
            "passed": 53,
            "skipped": 0,
            "failed": 0,
            "total": 53,
            "selected_total": 53,
            "complete": True,
            "pass_rate": 1.0,
        },
        "group_summary": groups,
        "cases": cases,
    }


class QualityComparisonTest(unittest.TestCase):

    def test_identical_qualified_reports_pass_quality_only(self):
        result = MODULE.compare_reports(
            make_report("baseline"), make_report("candidate"))
        self.assertTrue(result["qualified"])
        self.assertTrue(result["quality_non_regression_authorized"])
        self.assertFalse(result["overall_promotion_authorized"])
        self.assertEqual(result["summary"]["compared_cases"], 53)

    def test_exact_output_regression_fails(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        candidate["cases"][0]["observation"][
            "semantic_output_sha256"] = "c" * 64
        result = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "basic_chat: deterministic normalized output differs",
            result["reasons"],
        )

    def test_semantic_output_can_differ_after_independent_rule_passes(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        truncation = next(
            case for case in candidate["cases"]
            if case["id"] == "exact_output_truncation")
        truncation["observation"]["semantic_output_sha256"] = "d" * 64
        result = MODULE.compare_reports(baseline, candidate)
        self.assertTrue(result["qualified"])

    def test_tokenizer_or_protocol_regression_fails(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        candidate["cases"][0]["observation"]["prompt_tokens"] = [11]
        result = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "basic_chat: prompt tokenization differs", result["reasons"])

        candidate = make_report("candidate")
        candidate["tokenizer"]["chat_template_sha256"] = "0" * 64
        result = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "baseline and candidate tokenizer identities differ",
            result["reasons"],
        )

    def test_ab_may_only_change_declared_optimization_environment(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        contract = candidate["runtime_contract"]["contract"]
        contract["environment"]["BI100_GDN_CACHE_POLICY"] = "admission64"
        candidate["runtime_contract"]["sha256"] = (
            MODULE.runtime_contract.sha256_json(contract))
        candidate["runtime"]["service_env_sha256"] = (
            MODULE.runtime_contract.sha256_json(contract["environment"]))
        self.assertTrue(MODULE.compare_reports(
            baseline, candidate)["qualified"])

        candidate = make_report("candidate")
        contract = candidate["runtime_contract"]["contract"]
        contract["environment"]["BI100_UNDOCUMENTED_FAST_PATH"] = "1"
        candidate["runtime_contract"]["sha256"] = (
            MODULE.runtime_contract.sha256_json(contract))
        candidate["runtime"]["service_env_sha256"] = (
            MODULE.runtime_contract.sha256_json(contract["environment"]))
        self.assertFalse(MODULE.compare_reports(
            baseline, candidate)["qualified"])

    def test_ab_requires_same_source_overlay_and_instance(self):
        baseline = make_report("baseline")
        for field, value in (
                ("source_revision", "1" * 40),
                ("runtime_identity", "different-runtime"),
                ("runtime_overlay_sha256", "1" * 64),
                ("instance", "different-instance")):
            with self.subTest(field=field):
                candidate = make_report("candidate")
                candidate["runtime"][field] = value
                candidate["runtime_contract"]["contract"][field] = value
                candidate["runtime_contract"]["sha256"] = (
                    MODULE.runtime_contract.sha256_json(
                        candidate["runtime_contract"]["contract"]))
                self.assertFalse(MODULE.compare_reports(
                    baseline, candidate)["qualified"])

    def test_incomplete_or_raw_report_fails(self):
        baseline = make_report("baseline")
        candidate = copy.deepcopy(baseline)
        candidate["privacy"]["contains_raw_model_outputs"] = True
        candidate["summary"]["passed"] = 52
        result = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "candidate: privacy declaration is invalid", result["reasons"])

    def test_jointly_forged_case_identity_fails_canonical_manifest(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        for report in (baseline, candidate):
            report["cases"][0]["id"] = "forged_case"
        result = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "baseline: case basic_chat metadata differs", result["reasons"])

    def test_forged_summary_or_group_summary_fails(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        candidate["summary"]["passed"] = 52
        candidate["group_summary"]["basic"]["passed"] -= 1
        result = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "candidate: summary passed differs from cases", result["reasons"])
        self.assertIn(
            "candidate: group summary differs from cases", result["reasons"])

    def test_manifest_source_identity_cannot_be_self_reported(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        baseline["manifest"]["source_sha256"] = "e" * 64
        candidate["manifest"]["source_sha256"] = "e" * 64
        result = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "baseline: canonical manifest identity differs", result["reasons"])

    def test_only_explicit_direct_n2_skip_is_allowed(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        for report in (baseline, candidate):
            report["runtime"]["allow_bare_engine_n2_skip"] = True
            report["selection"]["allowed_skip_ids"] = ["n_2"]
            case = next(row for row in report["cases"] if row["id"] == "n_2")
            case["status"] = "skip"
            case["skip_reason"] = "documented bare-engine n=2 limitation"
            case["observation"]["status_codes"] = [400]
            case["observation"]["finish_reasons"] = []
            case["observation"]["prompt_tokens"] = []
            case["observation"]["cached_tokens"] = []
            case["observation"]["completion_tokens"] = []
            case["observation"]["facts"].update({
                "documented_bare_engine_skip": True,
                "normalized_error": "n_exceeds_max_num_seqs",
                "post_skip_health": True,
            })
            report["summary"]["passed"] = 52
            report["summary"]["skipped"] = 1
            report["group_summary"]["sampling"]["passed"] -= 1
            report["group_summary"]["sampling"]["skipped"] = 1
        result = MODULE.compare_reports(baseline, candidate)
        self.assertTrue(result["qualified"])

    def test_rejected_request_cannot_be_forged_as_http_200(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        for report in (baseline, candidate):
            case = next(
                row for row in report["cases"]
                if row["id"] == "empty_request_body")
            case["observation"].update({
                "status_codes": [200],
                "finish_reasons": ["stop"],
                "prompt_tokens": [1],
                "cached_tokens": [0],
                "completion_tokens": [1],
            })
        result = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "baseline: case empty_request_body: rejected request did not "
            "return one 4xx",
            result["reasons"],
        )

    def test_cache_and_truncation_evidence_cannot_be_forged(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        prefix = next(
            row for row in candidate["cases"]
            if row["id"] == "prefix_cache_hit")
        prefix["observation"]["cached_tokens"] = [0, 0]
        truncation = next(
            row for row in candidate["cases"]
            if row["id"] == "exact_output_truncation")
        truncation["observation"]["completion_tokens"] = [2]
        truncation["observation"]["facts"]["exact_completion_tokens"] = 2
        result = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "candidate: case prefix_cache_hit: prefix cold/warm accounting "
            "differs",
            result["reasons"],
        )
        self.assertIn(
            "candidate: case exact_output_truncation: exact truncation "
            "evidence differs",
            result["reasons"],
        )

    def test_max_tokens_one_accepts_length_but_rejects_overrun(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        self.assertTrue(MODULE.compare_reports(
            baseline, candidate)["qualified"])

        case = next(row for row in candidate["cases"]
                    if row["id"] == "max_tokens_1")
        case["observation"]["completion_tokens"] = [2]
        result = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "candidate: case max_tokens_1: max_tokens=1 enforcement "
            "evidence differs", result["reasons"])

    def test_cross_image_cached_tokens_must_be_zero(self):
        baseline = make_report("baseline")
        candidate = make_report("candidate")
        case = next(
            row for row in candidate["cases"]
            if row["id"] == "multimodal_input")
        case["observation"]["cached_tokens"] = [0, 48, 16]
        result = MODULE.compare_reports(baseline, candidate)
        self.assertFalse(result["qualified"])
        self.assertIn(
            "candidate: case multimodal_input: multimodal cache isolation "
            "differs",
            result["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
