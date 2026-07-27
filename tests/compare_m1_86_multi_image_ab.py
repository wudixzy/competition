#!/usr/bin/env python3
"""Qualify the fixed M1-86 single-GPU image-limit A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


Json = dict[str, Any]
SCHEMA = "bi100-m1-86-multi-image-ab-v1"
VERSION = 1
GATE_SCHEMA = "qwen36-diagnostic-multi-image-http-gate-v1"
STATUS_SCHEMA = "bi100-m1-86-multi-image-arm-v1"
CONTRACT_SCHEMA = "bi100-m1-86-service-contract-v1"
TRACE_SCHEMA = "bi100-m1-86-multi-image-trace-v1"
STATUS_ARTIFACT_NAMES = frozenset({
    "probe",
    "attribution",
    "capacity",
    "cache_trace",
    "service_contract",
    "process_group_identity",
    "service_postflight",
    "preflight_comparison",
})
ARM_GATE_NAMES = frozenset({
    "preflight_before",
    "port_preflight",
    "service_contract",
    "process_group",
    "startup",
    "probe",
    "capacity",
    "cleanup",
    "cache_trace",
    "attribution",
    "fatal_scan",
    "service_postflight",
    "preflight_after",
    "preflight_comparison",
})
CASE_NAMES = (
    "models_262144_contract",
    "stream_one_image_cold",
    "stream_two_images_cold",
    "stream_two_images_warm",
    "stream_two_images_reversed",
    "stream_two_images_reversed_warm",
    "post_request_health",
)
EXACT_FIELDS = (
    "semantic_output_sha256",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "has_content",
    "has_reasoning_content",
    "tool_call_count",
)
REFERENCE_ENVIRONMENT = {
    "BI100_ATTN_COREX_PAGED_GATHER": "1",
    "BI100_BLOCK_MAJOR_CPU_KV": "0",
    "BI100_CPU_KV_OFFLOAD": "0",
    "BI100_GDN_CACHE_POLICY": "fine32",
    "BI100_GDN_COMBINED_QK_NORM": "0",
    "BI100_GDN_COREX_PACKED_DECODE": "0",
    "BI100_GDN_RESTORE_MODE": "direct",
    "BI100_HYBRID_KV_ACCOUNTING": "full_attention",
    "BI100_MOE_COREX_DIRECT_ROUTED": "0",
    "BI100_PREFIX_DTYPE": "float16",
    "BI100_PREFIX_MODEL_FINGERPRINT":
        "Qwen3.6-35B-A3B-diagnostic-4L-real",
    "BI100_PREFIX_TP_SIZE": "1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _case_map(report: Json, label: str, reasons: list[str]) -> dict[str, Json]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        reasons.append(f"{label} cases are missing")
        return {}
    result: dict[str, Json] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            reasons.append(f"{label} contains a malformed case")
            continue
        if case["name"] in result:
            reasons.append(f"{label} duplicates case {case['name']}")
        result[case["name"]] = case
    if tuple(result) != CASE_NAMES:
        reasons.append(f"{label} case order or identity differs")
    return result


def _evidence(
    cases: dict[str, Json],
    name: str,
    label: str,
    reasons: list[str],
) -> Json:
    case = cases.get(name)
    if not isinstance(case, dict) or case.get("ok") is not True:
        reasons.append(f"{label} case {name} did not pass")
        return {}
    evidence = case.get("evidence")
    if not isinstance(evidence, dict):
        reasons.append(f"{label} case {name} has no evidence")
        return {}
    return evidence


def _normalize_command(
    command: Any,
    *,
    candidate: bool,
    reasons: list[str],
) -> list[str]:
    if not isinstance(command, list) or not all(
            isinstance(item, str) for item in command):
        reasons.append("service command is malformed")
        return []
    normalized = list(command)
    selector = "--limit-mm-per-prompt"
    indexes = [
        index for index, item in enumerate(normalized)
        if item == selector
    ]
    expected = [len(normalized) - 2] if candidate else []
    if indexes != expected:
        reasons.append(
            "candidate image-limit selector is not the sole command suffix"
            if candidate else
            "control command unexpectedly contains an image-limit selector")
        return normalized
    if candidate:
        if normalized[-1] != "image=2":
            reasons.append("candidate image limit is not exactly image=2")
        normalized = normalized[:-2]
    return normalized


def compare(
    control_report: Json,
    candidate_report: Json,
    control_attribution: Json,
    candidate_attribution: Json,
    control_status: Json,
    candidate_status: Json,
    control_contract: Json,
    candidate_contract: Json,
    control_capacity: Json,
    candidate_capacity: Json,
    control_trace: Json,
    candidate_trace: Json,
    control_process_group: Json,
    candidate_process_group: Json,
    artifact_sha256: dict[str, str],
) -> Json:
    reasons: list[str] = []
    required_artifact_inputs = frozenset({
        "control_report",
        "candidate_report",
        "control_attribution",
        "candidate_attribution",
        "control_status",
        "candidate_status",
        "control_contract",
        "candidate_contract",
        "control_capacity",
        "candidate_capacity",
        "control_trace",
        "candidate_trace",
        "control_process_group",
        "candidate_process_group",
    })
    if (
        set(artifact_sha256) != required_artifact_inputs
        or not all(_valid_sha256(value)
                   for value in artifact_sha256.values())
    ):
        reasons.append("aggregate artifact digest set differs")

    for label, report, expected_status in (
        ("control", control_report, 400),
        ("candidate", candidate_report, 200),
    ):
        if (
            report.get("schema") != GATE_SCHEMA
            or report.get("version") != 1
            or report.get("qualified") is not True
            or report.get("case_count") != len(CASE_NAMES)
        ):
            reasons.append(f"{label} HTTP gate is not qualified")
        if (
            report.get("config", {}).get("expected_two_image_status")
            != expected_status
        ):
            reasons.append(f"{label} expected image status differs")
        if (
            report.get("config", {}).get("stream") is not True
            or report.get("config", {}).get("temperature") != 0
            or report.get("config", {}).get("seed") != 20260728
            or report.get("config", {}).get("max_tokens") != 8
            or report.get("config", {}).get("thinking") is not False
        ):
            reasons.append(f"{label} deterministic streaming contract differs")
        privacy = report.get("privacy")
        if (
            not isinstance(privacy, dict)
            or privacy.get("contains_raw_request") is not False
            or privacy.get("contains_raw_response") is not False
            or privacy.get("contains_image_url_or_bytes") is not False
            or privacy.get("contains_prompt_or_generated_text") is not False
            or privacy.get("contains_credentials") is not False
            or privacy.get("synthetic_images_only") is not True
        ):
            reasons.append(f"{label} privacy contract differs")
        if (
            report.get("semantic_quality_evaluated") is not False
            or report.get("full_model_evaluated") is not False
            or report.get("production_promotion_authorized") is not False
        ):
            reasons.append(f"{label} diagnostic authority boundary differs")

    control_cases = _case_map(control_report, "control", reasons)
    candidate_cases = _case_map(candidate_report, "candidate", reasons)
    control_one = _evidence(
        control_cases, "stream_one_image_cold", "control", reasons)
    candidate_one = _evidence(
        candidate_cases, "stream_one_image_cold", "candidate", reasons)
    control_two = _evidence(
        control_cases, "stream_two_images_cold", "control", reasons)
    candidate_two = _evidence(
        candidate_cases, "stream_two_images_cold", "candidate", reasons)
    candidate_warm = _evidence(
        candidate_cases, "stream_two_images_warm", "candidate", reasons)
    candidate_reversed = _evidence(
        candidate_cases, "stream_two_images_reversed", "candidate", reasons)
    candidate_reversed_warm = _evidence(
        candidate_cases,
        "stream_two_images_reversed_warm",
        "candidate",
        reasons,
    )

    if control_two.get("http_status") != 400:
        reasons.append("control did not reject the second image")
    if candidate_two.get("http_status") != 200:
        reasons.append("candidate did not accept two images")
    if not all(
        control_one.get(field) == candidate_one.get(field)
        for field in EXACT_FIELDS
    ):
        reasons.append("one-image deterministic output differs across arms")
    if candidate_warm.get("cold_generation_exact") is not True:
        reasons.append("candidate two-image cold/warm output is not exact")
    if not all(
        candidate_two.get(field) == candidate_warm.get(field)
        for field in EXACT_FIELDS
    ):
        reasons.append("candidate two-image cold/warm summary differs")
    if not isinstance(candidate_warm.get("cached_tokens"), int) or (
        candidate_warm.get("cached_tokens", 0) <= 0
    ):
        reasons.append("candidate warm two-image request has no cache hit")
    if candidate_reversed.get("cache_isolation_deferred_to_trace") is not True:
        reasons.append("candidate reversed-image trace handoff is missing")
    if candidate_reversed_warm.get("cold_generation_exact") is not True:
        reasons.append("candidate reversed-image replay is not exact")
    if not all(
        candidate_reversed.get(field) == candidate_reversed_warm.get(field)
        for field in EXACT_FIELDS
    ):
        reasons.append("candidate reversed-image warm summary differs")
    if not isinstance(candidate_reversed_warm.get("cached_tokens"), int) or (
        candidate_reversed_warm.get("cached_tokens", 0) <= 0
    ):
        reasons.append("candidate reversed-image warm request has no cache hit")

    for label, trace, expected_count in (
        ("control", control_trace, 1),
        ("candidate", candidate_trace, 5),
    ):
        privacy = trace.get("privacy")
        if (
            trace.get("schema") != TRACE_SCHEMA
            or trace.get("version") != 1
            or trace.get("qualified") is not True
            or trace.get("mode") != label
            or trace.get("trace_version") != 4
            or trace.get("trace_count") != expected_count
            or not isinstance(privacy, dict)
            or privacy.get("contains_raw_tokens") is not False
            or privacy.get("contains_raw_images") is not False
            or privacy.get("contains_raw_prompt_or_output") is not False
            or privacy.get("contains_request_id") is not False
            or privacy.get("contains_credentials") is not False
            or trace.get("semantic_quality_evaluated") is not False
            or trace.get("production_promotion_authorized") is not False
        ):
            reasons.append(f"{label} cache trace did not qualify")

    process_group_pids: list[int] = []
    for label, identity in (
        ("control", control_process_group),
        ("candidate", candidate_process_group),
    ):
        pid = identity.get("pid")
        if (
            identity.get("schema") != "bi100-process-session-v1"
            or identity.get("version") != 1
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or identity.get("pgid") != pid
            or identity.get("sid") != pid
        ):
            reasons.append(f"{label} process session identity differs")
        else:
            process_group_pids.append(pid)
    if (
        len(process_group_pids) == 2
        and process_group_pids[0] == process_group_pids[1]
    ):
        reasons.append("service arms reused one process session")

    for label, attribution, expected_count in (
        ("control", control_attribution, 1),
        ("candidate", candidate_attribution, 0),
    ):
        if (
            attribution.get("schema") != "bi100-api-4xx-attribution-v3"
            or attribution.get("qualified") is not True
            or attribution.get("complete") is not True
            or attribution.get("classified") is not True
            or attribution.get("chat_4xx_access_count") != expected_count
            or attribution.get("attributed_count") != expected_count
            or attribution.get("attribution_delta") != 0
        ):
            reasons.append(f"{label} 4xx attribution is not complete")
    if control_attribution.get("by_reason") != {"image_count_limit": 1}:
        reasons.append("control 4xx reason is not exactly image_count_limit")
    if candidate_attribution.get("by_reason") != {}:
        reasons.append("candidate unexpectedly emitted a 4xx reason")
    shapes = control_attribution.get("request_shapes")
    if not isinstance(shapes, list) or len(shapes) != 1:
        reasons.append("control image-limit request shape is missing")
    else:
        shape = shapes[0]
        if (
            shape.get("count") != 1
            or shape.get("images") != 2
            or shape.get("image_data") != 2
            or shape.get("image_remote") != 0
            or shape.get("image_other") != 0
            or shape.get("stream") != 1
        ):
            reasons.append("control image-limit request shape differs")

    for label, status in (
        ("control", control_status),
        ("candidate", candidate_status),
    ):
        expected_image_limit = 1 if label == "control" else 2
        gates = status.get("gates")
        status_artifacts = status.get("artifact_sha256")
        if (
            status.get("schema") != STATUS_SCHEMA
            or status.get("version") != 1
            or status.get("qualified") is not True
            or status.get("returncode") != 0
            or status.get("image_limit") != expected_image_limit
            or not isinstance(gates, dict)
            or set(gates) != ARM_GATE_NAMES
            or any(value != 0 for value in gates.values())
            or not isinstance(status_artifacts, dict)
            or set(status_artifacts) != STATUS_ARTIFACT_NAMES
            or not all(_valid_sha256(value)
                       for value in status_artifacts.values())
        ):
            reasons.append(f"{label} arm lifecycle did not qualify")
            continue
        expected_status_artifacts = {
            "probe": artifact_sha256.get(f"{label}_report"),
            "attribution": artifact_sha256.get(f"{label}_attribution"),
            "capacity": artifact_sha256.get(f"{label}_capacity"),
            "cache_trace": artifact_sha256.get(f"{label}_trace"),
            "service_contract": artifact_sha256.get(f"{label}_contract"),
            "process_group_identity":
                artifact_sha256.get(f"{label}_process_group"),
        }
        if any(
            status_artifacts.get(name) != digest
            for name, digest in expected_status_artifacts.items()
        ):
            reasons.append(f"{label} arm artifact binding differs")

    for label, contract, image_limit in (
        ("control", control_contract, 1),
        ("candidate", candidate_contract, 2),
    ):
        if (
            contract.get("schema") != CONTRACT_SCHEMA
            or contract.get("version") != 1
            or contract.get("tensor_parallel_size") != 1
            or contract.get("max_model_len") != 262144
            or contract.get("image_limit") != image_limit
            or contract.get("runtime_source_files_match") is not True
            or contract.get("semantic_quality_evaluated") is not False
            or contract.get("production_promotion_authorized") is not False
        ):
            reasons.append(f"{label} service contract differs")
        environment = contract.get("environment")
        if (
            not isinstance(environment, dict)
            or any(environment.get(name) != expected
                   for name, expected in REFERENCE_ENVIRONMENT.items())
        ):
            reasons.append(f"{label} reference environment differs")
    control_command = _normalize_command(
        control_contract.get("command"), candidate=False, reasons=reasons)
    candidate_command = _normalize_command(
        candidate_contract.get("command"), candidate=True, reasons=reasons)
    if control_command != candidate_command:
        reasons.append("service commands differ outside the image limit")
    for field in (
        "source_revision",
        "source_branch",
        "runtime_tree_sha256",
        "runtime_install_sha256",
        "model_path",
        "model_manifest_sha256",
        "environment",
        "tensor_parallel_size",
        "max_model_len",
    ):
        if control_contract.get(field) != candidate_contract.get(field):
            reasons.append(f"service contracts differ at {field}")

    for label, capacity in (
        ("control", control_capacity),
        ("candidate", candidate_capacity),
    ):
        if (
            capacity.get("qualified") is not True
            or capacity.get("max_model_len_required") != 262144
            or capacity.get("required_gpu_blocks") != 16384
            or not isinstance(capacity.get("observed_gpu_blocks"), int)
            or capacity.get("observed_gpu_blocks", 0) < 16384
        ):
            reasons.append(f"{label} 262144 capacity did not qualify")
    control_blocks = control_capacity.get("observed_gpu_blocks")
    candidate_blocks = candidate_capacity.get("observed_gpu_blocks")
    block_ratio = (
        candidate_blocks / control_blocks
        if isinstance(control_blocks, int) and control_blocks > 0
        and isinstance(candidate_blocks, int)
        else None
    )
    if block_ratio is None or block_ratio < 0.98:
        reasons.append("candidate image budget loses more than 2% GPU blocks")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "observed": {
            "control_gpu_blocks": control_blocks,
            "candidate_gpu_blocks": candidate_blocks,
            "candidate_control_gpu_block_ratio": block_ratio,
            "one_image_generation_exact": not any(
                reason == "one-image deterministic output differs across arms"
                for reason in reasons
            ),
            "two_image_cold_warm_exact":
                candidate_warm.get("cold_generation_exact") is True,
            "two_image_warm_cached_tokens":
                candidate_warm.get("cached_tokens"),
            "reversed_image_initial_cached_tokens":
                candidate_reversed.get("cached_tokens"),
            "reversed_image_warm_cached_tokens":
                candidate_reversed_warm.get("cached_tokens"),
            "candidate_content_isolation":
                candidate_trace.get("content_isolation"),
            "control_4xx_reasons":
                control_attribution.get("by_reason"),
            "candidate_4xx_reasons":
                candidate_attribution.get("by_reason"),
        },
        "decision": {
            "single_gpu_diagnostic_phase_passed": not reasons,
            "full_model_tp4_required": True,
            "semantic_quality_required": True,
            "official_881_required": True,
            "main_or_yaml_change_authorized": False,
            "default_image_limit_change_authorized": False,
            "production_promotion_authorized": False,
        },
    }


def _atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
                      sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--control-attribution", type=Path, required=True)
    parser.add_argument("--candidate-attribution", type=Path, required=True)
    parser.add_argument("--control-status", type=Path, required=True)
    parser.add_argument("--candidate-status", type=Path, required=True)
    parser.add_argument("--control-contract", type=Path, required=True)
    parser.add_argument("--candidate-contract", type=Path, required=True)
    parser.add_argument("--control-capacity", type=Path, required=True)
    parser.add_argument("--candidate-capacity", type=Path, required=True)
    parser.add_argument("--control-trace", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--control-process-group", type=Path, required=True)
    parser.add_argument("--candidate-process-group", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = {
        name: value
        for name, value in vars(args).items()
        if name != "out"
    }
    values = {name: _load(path) for name, path in paths.items()}
    artifact_sha256 = {
        name: _sha256(path) for name, path in paths.items()
    }
    report = compare(**values, artifact_sha256=artifact_sha256)
    report["artifact_sha256"] = artifact_sha256
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
