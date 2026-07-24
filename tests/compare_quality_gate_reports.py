#!/usr/bin/env python3
"""Compare a candidate quality run with a frozen CoreX baseline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "quality/official_metrics_manifest.v1.json"
REPORT_SCHEMA = "bi100-quality-gate-result-v1"
COMPARISON_SCHEMA = "bi100-quality-baseline-comparison-v1"
EXPECTED_SOURCE_SHA256 = (
    "116e7edc617d8f96fc92caa3e75a3ba4692aae7619026896df1eaf69df12feac"
)
EXPECTED_MANIFEST_SHA256 = (
    "fe9b958610d9d0df8f54504d9c149442f145226c03cf76668711d2d38ed51d0e"
)
EXPECTED_CASES = 53
Json = dict[str, Any]

ALWAYS_REJECTED = {
    "top_p_1_1",
    "max_tokens_minus_1",
    "max_tokens_over_context",
    "empty_request_body",
    "message_missing_role",
    "message_missing_content",
    "empty_messages",
}
MULTI_RESPONSE_COUNTS = {
    "multimodal_input": 3,
    "base64_png": 3,
    "prefix_cache_hit": 2,
    "idempotency": 2,
}
TRUE_FACTS = {
    "basic_chat": ("schema_content_usage_valid",),
    "streaming_usage": (
        "completion_tokens_positive", "ten_distinct_color_lines_exact",
        "incremental_events_observed",
    ),
    "streaming_sse_usage": (
        "completion_tokens_positive", "ten_distinct_color_lines_exact",
        "incremental_events_observed",
    ),
    "tool_calling": ("arguments_valid_json",),
    "function_calling": ("arguments_valid_json",),
    "reasoning": ("answer_rule_passed",),
    "multimodal_input": (
        "content_length_gt_15", "red_identified", "blue_identified",
        "same_image_cold_warm_exact", "different_image_isolated",
        "cold_and_cross_image_cached_tokens_zero",
    ),
    "base64_png": (
        "content_length_gt_15", "red_identified", "blue_identified",
        "same_image_cold_warm_exact", "different_image_isolated",
        "cold_and_cross_image_cached_tokens_zero",
    ),
    "prefix_cache_hit": (
        "cold_warm_exact", "cold_cached_tokens_zero",
        "warm_cached_tokens_positive",
    ),
    "reasoning_content_split": (
        "reasoning_content_nonempty", "content_nonempty",
    ),
    "max_tokens_unset": ("natural_stop",),
    "max_tokens_1": ("natural_stop",),
    "max_tokens_64": ("natural_stop",),
    "max_tokens_64k": ("natural_stop",),
    "max_tokens_near_context": ("natural_stop",),
    "max_tokens_minus_1": ("rejected_without_5xx",),
    "max_tokens_over_context": ("rejected_without_5xx",),
    "no_system_prompt": ("exact_echo",),
    "system_prompt_effective": ("exact_echo",),
    "multi_turn_memory": ("memory_rule_passed",),
    "json_object": ("valid_json", "values_match"),
    "json_schema": ("schema_valid", "values_match"),
    "stop_sequence": ("pre_stop_present", "post_stop_absent"),
    "chinese": ("exact_echo",),
    "japanese": ("exact_echo",),
    "emoji": ("exact_echo",),
    "empty_request_body": ("rejected_without_5xx",),
    "idempotency": ("deterministic",),
    "message_missing_role": ("rejected_without_5xx",),
    "message_missing_content": ("rejected_without_5xx",),
    "empty_messages": ("rejected_without_5xx",),
}
PARAMETER_FACTS = {
    "temperature_0": ("temperature", True),
    "temperature_1": ("temperature", True),
    "temperature_1_1": ("temperature", True),
    "temperature_2": ("temperature", True),
    "top_p_0": ("top_p", None),
    "top_p_0_01": ("top_p", True),
    "top_p_0_95": ("top_p", True),
    "top_p_1": ("top_p", True),
    "top_p_1_1": ("top_p", False),
    "frequency_penalty_minus_2": ("frequency_penalty", True),
    "frequency_penalty_0": ("frequency_penalty", True),
    "frequency_penalty_2": ("frequency_penalty", True),
    "presence_penalty_minus_2": ("presence_penalty", True),
    "presence_penalty_0": ("presence_penalty", True),
    "presence_penalty_2": ("presence_penalty", True),
}
MAX_TOKEN_FACTS = {
    "max_tokens_unset": None,
    "max_tokens_1": 1,
    "max_tokens_64": 64,
    "max_tokens_64k": 65536,
    "max_tokens_near_context": 261120,
    "max_tokens_minus_1": -1,
    "max_tokens_over_context": 262145,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_manifest(path: Path) -> tuple[Json, str]:
    payload = path.read_bytes()
    payload_sha = hashlib.sha256(payload).hexdigest()
    if payload_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError("canonical quality manifest SHA-256 differs")
    manifest = json.loads(payload)
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict)
            or manifest.get("schema") != "bi100-quality-metric-manifest-v1"
            or manifest.get("version") != 1
            or (manifest.get("source") or {}).get("sha256")
            != EXPECTED_SOURCE_SHA256
            or manifest.get("allowed_skips") != {"direct": ["n_2"]}
            or not isinstance(cases, list)
            or len(cases) != EXPECTED_CASES):
        raise ValueError("canonical quality manifest is invalid")
    metadata = ("ordinal", "id", "group", "tier", "comparison")
    identities = []
    for index, case in enumerate(cases, 1):
        if (not isinstance(case, dict) or case.get("ordinal") != index
                or any(key not in case for key in metadata)):
            raise ValueError("canonical quality manifest case is invalid")
        identities.append(case["id"])
    if len(set(identities)) != EXPECTED_CASES:
        raise ValueError("canonical quality manifest ids are not unique")
    return manifest, payload_sha


def _report_reasons(
    report: Any,
    label: str,
    manifest: Json,
    manifest_sha: str,
    manifest_name: str,
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(report, dict):
        return [f"{label}: report root must be an object"]
    if report.get("schema") != REPORT_SCHEMA or report.get("version") != 1:
        reasons.append(f"{label}: report schema or version is invalid")
    if report.get("qualified") is not True:
        reasons.append(f"{label}: quality run is not qualified")
    if report.get("quality_run_eligible_for_baseline") is not True:
        reasons.append(f"{label}: run is not eligible for baseline comparison")
    if report.get("promotion_authorized") is not False:
        reasons.append(f"{label}: standalone run must not authorize promotion")

    expected_manifest = {
        "path_name": manifest_name,
        "sha256": manifest_sha,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "total_cases": EXPECTED_CASES,
    }
    if report.get("manifest") != expected_manifest:
        reasons.append(f"{label}: canonical manifest identity differs")

    runtime = report.get("runtime") or {}
    required_runtime_strings = (
        "source_revision", "runtime_identity", "instance", "model_path",
        "tokenizer_path",
    )
    if any(not isinstance(runtime.get(key), str) or not runtime[key]
           for key in required_runtime_strings):
        reasons.append(f"{label}: runtime identity is incomplete")
    endpoint_mode = runtime.get("endpoint_mode")
    allow_skip = runtime.get("allow_bare_engine_n2_skip")
    if (runtime.get("max_model_len") != 262144
            or runtime.get("model") != "llm"
            or not isinstance(runtime.get("gpu_count"), int)
            or isinstance(runtime.get("gpu_count"), bool)
            or runtime.get("gpu_count", 0) <= 0
            or endpoint_mode not in ("direct", "gateway")
            or not isinstance(allow_skip, bool)
            or (allow_skip and endpoint_mode != "direct")):
        reasons.append(f"{label}: runtime capacity or topology is invalid")

    expected_skips = ["n_2"] if allow_skip else []
    selection = report.get("selection") or {}
    if (selection.get("tier") != "extended"
            or selection.get("explicit_cases") != []
            or selection.get("selected_cases") != EXPECTED_CASES
            or selection.get("allowed_skip_ids") != expected_skips):
        reasons.append(f"{label}: report is not the complete extended tier")

    privacy = report.get("privacy") or {}
    if (privacy.get("contains_raw_requests") is not False
            or privacy.get("contains_raw_model_outputs") is not False
            or privacy.get("contains_credentials") is not False):
        reasons.append(f"{label}: privacy declaration is invalid")
    return reasons


def _validate_observation(observation: Any, label: str) -> list[str]:
    if not isinstance(observation, dict):
        return [f"{label}: observation is not an object"]
    required = {
        "status_codes", "finish_reasons", "prompt_tokens", "cached_tokens",
        "completion_tokens", "semantic_output_sha256", "facts",
    }
    reasons = []
    if set(observation) != required:
        reasons.append(f"{label}: observation fields are invalid")
        return reasons
    status_codes = observation["status_codes"]
    if (not isinstance(status_codes, list) or not status_codes
            or any(not isinstance(value, int) or isinstance(value, bool)
                   or not 100 <= value <= 599 for value in status_codes)):
        reasons.append(f"{label}: status codes are invalid")
    if (not isinstance(observation["finish_reasons"], list)
            or any(value is not None and not isinstance(value, str)
                   for value in observation["finish_reasons"])):
        reasons.append(f"{label}: finish reasons are invalid")
    for field in ("prompt_tokens", "cached_tokens", "completion_tokens"):
        values = observation[field]
        if (not isinstance(values, list)
                or any(not isinstance(value, int) or isinstance(value, bool)
                       or value < 0 for value in values)):
            reasons.append(f"{label}: {field} are invalid")
    successful_responses = (
        status_codes.count(200) if isinstance(status_codes, list) else -1)
    token_lists = (
        observation["prompt_tokens"],
        observation["cached_tokens"],
        observation["completion_tokens"],
    )
    if all(isinstance(values, list) for values in token_lists):
        if any(len(values) != successful_responses for values in token_lists):
            reasons.append(f"{label}: usage count differs from HTTP 200 count")
        elif any(cached > prompt for prompt, cached in zip(
                observation["prompt_tokens"], observation["cached_tokens"])):
            reasons.append(f"{label}: cached tokens exceed prompt tokens")
    if not _is_sha256(observation["semantic_output_sha256"]):
        reasons.append(f"{label}: semantic output digest is invalid")
    if not isinstance(observation["facts"], dict):
        reasons.append(f"{label}: facts are invalid")
    return reasons


def _validate_case_contract(case: Json, report: Json, label: str) -> list[str]:
    case_id = case["id"]
    observation = case.get("observation") or {}
    statuses = observation.get("status_codes") or []
    facts = observation.get("facts") or {}
    reasons = []

    if case_id in ALWAYS_REJECTED:
        if len(statuses) != 1 or not 400 <= statuses[0] < 500:
            reasons.append(f"{label}: rejected request did not return one 4xx")
    elif case_id == "top_p_0":
        if len(statuses) != 1 or not (
                statuses[0] == 200 or 400 <= statuses[0] < 500):
            reasons.append(f"{label}: top_p=0 returned neither 200 nor 4xx")
    elif case_id == "thinking_disabled_top_level":
        valid = statuses == [200] or (
            len(statuses) == 2
            and 400 <= statuses[0] < 500
            and statuses[1] == 200
        )
        if not valid:
            reasons.append(f"{label}: thinking fallback status sequence is invalid")
    elif case_id == "n_2" and case.get("status") == "skip":
        if statuses != [400]:
            reasons.append(f"{label}: n=2 skip did not preserve the known 400")
    else:
        expected_count = MULTI_RESPONSE_COUNTS.get(case_id, 1)
        if statuses != [200] * expected_count:
            reasons.append(f"{label}: successful response count/status differs")

    for fact in TRUE_FACTS.get(case_id, ()):
        if facts.get(fact) is not True:
            reasons.append(f"{label}: required fact {fact} is not true")

    if case_id in PARAMETER_FACTS:
        parameter, accepted = PARAMETER_FACTS[case_id]
        if facts.get("parameter") != parameter:
            reasons.append(f"{label}: parameter fact differs")
        if accepted is None:
            if not isinstance(facts.get("accepted"), bool):
                reasons.append(f"{label}: parameter acceptance is not boolean")
        elif facts.get("accepted") is not accepted:
            reasons.append(f"{label}: parameter acceptance differs")

    if case_id in MAX_TOKEN_FACTS:
        if facts.get("requested_max_tokens", object()) != MAX_TOKEN_FACTS[case_id]:
            reasons.append(f"{label}: requested max_tokens fact differs")

    if case_id in ("tool_calling", "function_calling"):
        if (not isinstance(facts.get("tool_calls"), int)
                or isinstance(facts.get("tool_calls"), bool)
                or facts["tool_calls"] <= 0):
            reasons.append(f"{label}: tool call count is not positive")

    thinking_expected = {
        "thinking_disabled_top_level": (False, False),
        "thinking_true": (True, True),
        "thinking_false": (False, False),
        "thinking_default": (True, True),
    }
    if case_id in thinking_expected:
        enabled, reasoning = thinking_expected[case_id]
        if (facts.get("thinking_enabled") is not enabled
                or facts.get("reasoning_present") is not reasoning):
            reasons.append(f"{label}: thinking facts differ")
    if case_id == "thinking_disabled_top_level":
        protocol = facts.get("request_protocol")
        if protocol not in ("top_level", "direct_chat_template_fallback"):
            reasons.append(f"{label}: thinking request protocol is invalid")

    if case_id == "reasoning" and not isinstance(
            facts.get("reasoning_present"), bool):
        reasons.append(f"{label}: reasoning presence fact is not boolean")
    if case_id == "n_1" and facts.get("n") != 1:
        reasons.append(f"{label}: n=1 fact differs")
    if case_id == "n_2":
        if facts.get("n") != 2:
            reasons.append(f"{label}: n=2 fact differs")
        if case.get("status") == "skip" and (
                facts.get("documented_bare_engine_skip") is not True
                or facts.get("normalized_error") != "n_exceeds_max_num_seqs"
                or facts.get("post_skip_health") is not True):
            reasons.append(f"{label}: n=2 skip evidence differs")

    prompt = observation.get("prompt_tokens") or []
    cached = observation.get("cached_tokens") or []
    completion = observation.get("completion_tokens") or []
    finish = observation.get("finish_reasons") or []
    if case_id == "prefix_cache_hit":
        if not len(prompt) == len(cached) == len(completion) == 2:
            reasons.append(f"{label}: prefix usage shape differs")
        elif (cached[0] != 0 or cached[1] <= 0
                or prompt[0] != prompt[1]
                or completion[0] != completion[1]):
            reasons.append(f"{label}: prefix cold/warm accounting differs")
    if case_id in ("multimodal_input", "base64_png"):
        if not len(prompt) == len(cached) == len(completion) == 3:
            reasons.append(f"{label}: multimodal usage shape differs")
        elif (cached[0] != 0 or cached[1] <= 0 or cached[2] != 0
                or prompt[0] != prompt[1]
                or completion[0] != completion[1]):
            reasons.append(f"{label}: multimodal cache isolation differs")
    if case_id == "idempotency":
        if not len(prompt) == len(completion) == 2:
            reasons.append(f"{label}: idempotency usage shape differs")
        elif prompt[0] != prompt[1] or completion[0] != completion[1]:
            reasons.append(f"{label}: idempotency usage differs")
    if case_id == "exact_output_truncation":
        if (completion != [32768] or finish != ["length"]
                or facts.get("exact_completion_tokens") != 32768):
            reasons.append(f"{label}: exact truncation evidence differs")
    if case_id in (
            "max_tokens_unset", "max_tokens_1", "max_tokens_64",
            "max_tokens_64k", "max_tokens_near_context",
    ) and finish != ["stop"]:
        reasons.append(f"{label}: accepted max_tokens did not finish by stop")
    if case_id in ("streaming_usage", "streaming_sse_usage"):
        if (finish != ["stop"] or facts.get("done") != 1
                or facts.get("usage_blocks") != 1
                or not isinstance(facts.get("chunks"), int)
                or facts["chunks"] < 5):
            reasons.append(f"{label}: streaming protocol facts differ")
    return reasons


def _case_map(
    report: Json,
    label: str,
    manifest: Json,
) -> tuple[dict[str, Json], list[str]]:
    reasons: list[str] = []
    result: dict[str, Json] = {}
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        return {}, [f"{label}: report must contain 53 case results"]
    metadata = ("ordinal", "id", "group", "tier", "comparison")
    endpoint_mode = (report.get("runtime") or {}).get("endpoint_mode")
    allow_skip = (report.get("runtime") or {}).get(
        "allow_bare_engine_n2_skip") is True
    for expected, case in zip(manifest["cases"], cases):
        case_id = expected["id"]
        if not isinstance(case, dict):
            reasons.append(f"{label}: case {case_id} is not an object")
            continue
        if any(case.get(key) != expected.get(key) for key in metadata):
            reasons.append(f"{label}: case {case_id} metadata differs")
        status = case.get("status")
        skip_reason = case.get("skip_reason")
        error_code = case.get("error_code")
        if status not in ("pass", "skip", "fail"):
            reasons.append(f"{label}: case {case_id} status is invalid")
        expected_ok = status in ("pass", "skip")
        if case.get("ok") is not expected_ok:
            reasons.append(f"{label}: case {case_id} ok/status differ")
        if (not isinstance(case.get("elapsed_s"), (int, float))
                or isinstance(case.get("elapsed_s"), bool)
                or case.get("elapsed_s", -1) < 0):
            reasons.append(f"{label}: case {case_id} elapsed time is invalid")
        if not isinstance(error_code, str) or (expected_ok and error_code):
            reasons.append(f"{label}: case {case_id} error code is invalid")
        if status == "skip":
            if (case_id != "n_2" or endpoint_mode != "direct"
                    or not allow_skip or not isinstance(skip_reason, str)
                    or not skip_reason):
                reasons.append(f"{label}: case {case_id} skip is not allowed")
        elif skip_reason != "":
            reasons.append(f"{label}: case {case_id} has a stray skip reason")
        reasons.extend(_validate_observation(
            case.get("observation"), f"{label}: case {case_id}"))
        reasons.extend(_validate_case_contract(
            case, report, f"{label}: case {case_id}"))
        result[case_id] = case
    return result, reasons


def _recomputed_summary(report: Json) -> tuple[Json, dict[str, Json]]:
    cases = report.get("cases") or []
    statuses = [case.get("status") for case in cases if isinstance(case, dict)]
    passed = statuses.count("pass")
    skipped = statuses.count("skip")
    failed = statuses.count("fail")
    total = len(statuses)
    summary = {
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "total": total,
        "selected_total": EXPECTED_CASES,
        "complete": total == EXPECTED_CASES,
        "pass_rate": (passed + skipped) / total if total else 0.0,
    }
    groups: dict[str, Json] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        group = groups.setdefault(case.get("group"), {
            "passed": 0, "skipped": 0, "failed": 0, "total": 0,
        })
        group["total"] += 1
        status = case.get("status")
        if status in ("pass", "skip", "fail"):
            status_field = {
                "pass": "passed", "skip": "skipped", "fail": "failed",
            }[status]
            group[status_field] += 1
    for group in groups.values():
        group["pass_rate"] = (
            group["passed"] + group["skipped"]) / group["total"]
    return summary, groups


def _validate_recomputed(report: Json, label: str) -> list[str]:
    reasons = []
    expected_summary, expected_groups = _recomputed_summary(report)
    summary = report.get("summary") or {}
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            reasons.append(f"{label}: summary {key} differs from cases")
    if report.get("group_summary") != expected_groups:
        reasons.append(f"{label}: group summary differs from cases")
    if expected_summary["failed"] != 0:
        reasons.append(f"{label}: at least one case failed")
    return reasons


def compare_reports(
    baseline: Any,
    candidate: Any,
    *,
    manifest: Json | None = None,
    manifest_sha: str | None = None,
    manifest_name: str | None = None,
) -> Json:
    if manifest is None or manifest_sha is None:
        manifest, manifest_sha = _load_manifest(DEFAULT_MANIFEST)
        manifest_name = DEFAULT_MANIFEST.name
    assert manifest_name is not None
    reasons = _report_reasons(
        baseline, "baseline", manifest, manifest_sha, manifest_name)
    reasons.extend(_report_reasons(
        candidate, "candidate", manifest, manifest_sha, manifest_name))
    case_results: list[Json] = []

    if isinstance(baseline, dict) and isinstance(candidate, dict):
        baseline_runtime = baseline.get("runtime") or {}
        candidate_runtime = candidate.get("runtime") or {}
        for field in (
            "gpu_count", "model_path", "tokenizer_path", "max_model_len",
            "model", "endpoint_mode", "allow_bare_engine_n2_skip",
        ):
            if baseline_runtime.get(field) != candidate_runtime.get(field):
                reasons.append(f"runtime contract differs in {field}")
        baseline_cases, case_reasons = _case_map(
            baseline, "baseline", manifest)
        reasons.extend(case_reasons)
        candidate_cases, case_reasons = _case_map(
            candidate, "candidate", manifest)
        reasons.extend(case_reasons)
        reasons.extend(_validate_recomputed(baseline, "baseline"))
        reasons.extend(_validate_recomputed(candidate, "candidate"))

        for expected in manifest["cases"]:
            case_id = expected["id"]
            if case_id not in baseline_cases or case_id not in candidate_cases:
                continue
            base_case = baseline_cases[case_id]
            cand_case = candidate_cases[case_id]
            case_failures = []
            if base_case.get("status") != cand_case.get("status"):
                case_failures.append("pass/skip status differs")
            base_observation = base_case.get("observation") or {}
            cand_observation = cand_case.get("observation") or {}
            if base_observation.get("status_codes") != cand_observation.get(
                    "status_codes"):
                case_failures.append("HTTP status sequence differs")
            if base_observation.get("finish_reasons") != cand_observation.get(
                    "finish_reasons"):
                case_failures.append("finish_reason sequence differs")
            if base_observation.get("prompt_tokens") != cand_observation.get(
                    "prompt_tokens"):
                case_failures.append("prompt tokenization differs")
            if expected["comparison"] == "exact":
                if base_observation.get(
                        "semantic_output_sha256") != cand_observation.get(
                            "semantic_output_sha256"):
                    case_failures.append("deterministic normalized output differs")
                if base_observation.get(
                        "completion_tokens") != cand_observation.get(
                            "completion_tokens"):
                    case_failures.append("deterministic completion usage differs")
            elif expected["comparison"] == "semantic":
                if base_observation.get("facts") != cand_observation.get("facts"):
                    case_failures.append("independent semantic facts differ")
            case_results.append({
                "ordinal": expected["ordinal"],
                "id": case_id,
                "comparison": expected["comparison"],
                "qualified": not case_failures,
                "reasons": case_failures,
            })
            reasons.extend(f"{case_id}: {reason}" for reason in case_failures)

    qualified = not reasons and len(case_results) == EXPECTED_CASES
    return {
        "schema": COMPARISON_SCHEMA,
        "version": 1,
        "qualified": qualified,
        "quality_non_regression_authorized": qualified,
        "overall_promotion_authorized": False,
        "reasons": reasons,
        "summary": {
            "compared_cases": len(case_results),
            "qualified_cases": sum(row["qualified"] for row in case_results),
            "failed_cases": sum(not row["qualified"] for row in case_results),
        },
        "cases": case_results,
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_credentials": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest, manifest_sha = _load_manifest(args.manifest)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_reports(
        baseline,
        candidate,
        manifest=manifest,
        manifest_sha=manifest_sha,
        manifest_name=args.manifest.name,
    )
    result["inputs"] = {
        "manifest_file_sha256": manifest_sha,
        "baseline_file_sha256": _sha256(args.baseline),
        "candidate_file_sha256": _sha256(args.candidate),
        "baseline_label": baseline.get("label"),
        "candidate_label": candidate.get("label"),
        "baseline_source_revision": (baseline.get("runtime") or {}).get(
            "source_revision"),
        "candidate_source_revision": (candidate.get("runtime") or {}).get(
            "source_revision"),
    }
    _atomic_write(args.out, result)
    print(json.dumps({
        "qualified": result["qualified"],
        "reasons": result["reasons"],
        "summary": result["summary"],
        "out": str(args.out),
    }, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
