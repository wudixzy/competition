#!/usr/bin/env python3
"""Bind M1-89 runtime, M1-84 functional, and M1-86 image gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


Json = dict[str, Any]
SCHEMA = "bi100-m1-87-single-gpu-queue-v2"
CACHE_NAMESPACE_SCHEMA = "qwen36-cache-namespace-runtime-gate-v2"
DIAGNOSTIC_SCHEMA = "qwen36-diagnostic-service-gate-v1"
IMAGE_RUNNER_SCHEMA = "bi100-m1-86-multi-image-ab-runner-v1"
IMAGE_COMPARISON_SCHEMA = "bi100-m1-86-multi-image-ab-v1"
OVERLAY_SCHEMA = "bi100-bare-host-runtime-identity-v1"
POSTFLIGHT_SCHEMA = "bi100-service-postflight-v1"
RECOVERY_SCHEMA = "bi100-recorded-session-cleanup-v1"
SESSION_SCHEMA = "bi100-process-session-v1"
CACHE_NAMESPACE_CHECKS = frozenset({
    "module_bound_to_overlay",
    "same_palette_stable",
    "different_palette_isolated",
    "different_transparency_isolated",
    "empty_multimodal_matches_text",
    "truthiness_not_evaluated",
    "normalization_error_is_request_local",
    "release_clears_request_state",
    "request_id_reuse_gets_fresh_namespace",
})
DIAGNOSTIC_GATE_NAMES = frozenset({
    "checkpoint_verify",
    "runtime_overlay_identity",
    "runtime_identity",
    "preflight_before",
    "nccl_before",
    "port_preflight",
    "gdn_action_broadcast",
    "service_contract",
    "process_group",
    "startup",
    "api",
    "quality_contract",
    "compat_http",
    "tool_http",
    "prefix_boundary",
    "cleanup",
    "cleanup_status",
    "service_postflight",
    "fatal_scan",
    "timeout_scan",
    "layer_trace",
    "preflight_after",
    "preflight_comparison",
})
DIAGNOSTIC_ARTIFACT_NAMES = frozenset({
    "checkpoint_verify.json",
    "runtime_overlay_identity.json",
    "runtime_identity.json",
    "preflight_before.json",
    "nccl_before.json",
    "gdn_action_broadcast.json",
    "service_contract.json",
    "process_group_identity.json",
    "startup.json",
    "api_gate.json",
    "quality_contract_gate.json",
    "compat_http_gate.json",
    "tool_http_gate.json",
    "prefix_boundary.json",
    "server.log",
    "cleanup_status.json",
    "service_postflight.json",
    "preflight_after.json",
    "preflight_comparison.json",
})
IMAGE_RUNNER_GATE_NAMES = frozenset({
    "checkpoint_verify",
    "runtime_overlay_identity",
    "initial_preflight",
    "control",
    "candidate",
    "comparison",
    "cleanup",
    "final_postflight",
    "final_preflight",
    "final_preflight_comparison",
    "fatal_scan",
    "timeout_scan",
})
IMAGE_RUNNER_ARTIFACTS = {
    "runtime_overlay_identity": "runtime_overlay_identity.json",
    "comparison": "comparison.json",
    "final_postflight": "final_postflight.json",
    "final_preflight_comparison": "final_preflight_comparison.json",
}
IMAGE_COMPARISON_ARTIFACTS = {
    "control_report": "control/probe.json",
    "candidate_report": "candidate/probe.json",
    "control_attribution": "control/attribution.json",
    "candidate_attribution": "candidate/attribution.json",
    "control_status": "control/status.json",
    "candidate_status": "candidate/status.json",
    "control_contract": "control/service_contract.json",
    "candidate_contract": "candidate/service_contract.json",
    "control_capacity": "control/capacity.json",
    "candidate_capacity": "candidate/capacity.json",
    "control_trace": "control/cache_trace.json",
    "candidate_trace": "candidate/cache_trace.json",
    "control_startup": "control/startup.json",
    "candidate_startup": "candidate/startup.json",
    "control_process_group": "control/process_group_identity.json",
    "candidate_process_group": "candidate/process_group_identity.json",
    "control_postflight": "control/service_postflight.json",
    "candidate_postflight": "candidate/service_postflight.json",
    "control_preflight_comparison": "control/preflight_comparison.json",
    "candidate_preflight_comparison": "candidate/preflight_comparison.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_confined_regular_file(path: Path, root: Path) -> bool:
    try:
        if root.is_symlink():
            return False
        relative = path.relative_to(root)
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return False
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        return (
            resolved_path.is_relative_to(resolved_root)
            and resolved_path.is_file()
        )
    except (OSError, ValueError):
        return False


def _load(
    path: Path,
    label: str,
    reasons: list[str],
    root: Path,
) -> Json:
    if not _is_confined_regular_file(path, root):
        reasons.append(f"{label} is not a confined regular file")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reasons.append(f"{label} is missing or invalid: {type(exc).__name__}")
        return {}
    if not isinstance(value, dict):
        reasons.append(f"{label} must contain a JSON object")
        return {}
    return value


def _read_rc(
    path: Path,
    label: str,
    reasons: list[str],
    root: Path,
) -> int | None:
    if not _is_confined_regular_file(path, root):
        reasons.append(f"{label} rc is not a confined regular file")
        return None
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        reasons.append(f"{label} rc is missing or invalid: {type(exc).__name__}")
        return None
    if value != 0:
        reasons.append(f"{label} returned {value}")
    return value


def _exact_zero_gates(value: Any, expected: frozenset[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == expected
        and all(item == 0 for item in value.values())
    )


def _authority_is_diagnostic(report: Json) -> bool:
    return (
        report.get("full_model_evaluated") is False
        and report.get("semantic_quality_evaluated") is False
        and report.get("performance_evaluated") is False
        and report.get("production_promotion_authorized") is False
    )


def _check_artifact_manifest(
    report: Json,
    root: Path,
    label: str,
    reasons: list[str],
    expected: dict[str, str],
) -> None:
    artifacts = report.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected):
        reasons.append(f"{label} artifact binding is missing")
        return
    resolved_root = root.resolve()
    for name, relative in expected.items():
        path = root / relative
        digest = artifacts.get(name)
        try:
            resolved_path = path.resolve(strict=True)
            confined = resolved_path.is_relative_to(resolved_root)
        except OSError:
            confined = False
        if (
            not isinstance(name, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or not _is_confined_regular_file(path, root)
            or not confined
            or _sha256(path) != digest
        ):
            reasons.append(f"{label} artifact binding differs: {name}")


def qualify(
    root: Path,
    *,
    expected_source_revision: str,
    expected_gpu: int,
    runner_returncode: int,
) -> Json:
    reasons: list[str] = []
    diagnostic_root = root / "m1_84"
    image_root = root / "m1_86"

    rc_names = (
        "m1_89_overlay_identity",
        "m1_89_runtime_gate",
        "m1_84",
        "interstage_postflight",
        "interstage_preflight",
        "m1_86",
        "final_postflight",
        "final_preflight",
        "fatal_scan",
        "timeout_scan",
        "child_cleanup",
        "service_recovery",
    )
    gates = {
        name: _read_rc(root / f"{name}.rc", name, reasons, root)
        for name in rc_names
    }
    if runner_returncode != 0:
        reasons.append(f"queue primary return code is {runner_returncode}")

    cache_overlay = _load(
        root / "m1_89_runtime_overlay_identity.json",
        "M1-89 overlay identity",
        reasons,
        root,
    )
    cache_gate = _load(
        root / "m1_89_cache_namespace_runtime_gate.json",
        "M1-89 cache namespace runtime gate",
        reasons,
        root,
    )
    diagnostic = _load(
        diagnostic_root / "status.json", "M1-84 status", reasons, root)
    diagnostic_overlay = _load(
        diagnostic_root / "runtime_overlay_identity.json",
        "M1-84 overlay identity",
        reasons,
        root,
    )
    image_runner = _load(
        image_root / "runner_status.json", "M1-86 runner status", reasons, root)
    image_comparison = _load(
        image_root / "comparison.json", "M1-86 comparison", reasons, root)
    image_overlay = _load(
        image_root / "runtime_overlay_identity.json",
        "M1-86 overlay identity",
        reasons,
        root,
    )
    control_contract = _load(
        image_root / "control" / "service_contract.json",
        "M1-86 control contract",
        reasons,
        root,
    )
    candidate_contract = _load(
        image_root / "candidate" / "service_contract.json",
        "M1-86 candidate contract",
        reasons,
        root,
    )
    interstage_postflight = _load(
        root / "interstage_postflight.json", "interstage postflight", reasons,
        root)
    interstage_preflight = _load(
        root / "interstage_preflight.json", "interstage preflight", reasons,
        root)
    final_postflight = _load(
        root / "final_postflight.json", "final postflight", reasons, root)
    final_preflight = _load(
        root / "final_preflight.json", "final preflight", reasons, root)
    recovery = _load(
        root / "service_recovery.json", "service recovery", reasons, root)
    child_identities = [
        _load(
            root / f"{name}_child_identity.json",
            f"{name} child identity",
            reasons,
            root,
        )
        for name in ("m1_84", "m1_86")
    ]

    cache_checks = cache_gate.get("checks")
    cache_privacy = cache_gate.get("privacy")
    cache_module_digests = (
        cache_gate.get("block_manager_module_sha256"),
        cache_gate.get("sequence_module_sha256"),
    )
    if (
        cache_gate.get("schema") != CACHE_NAMESPACE_SCHEMA
        or cache_gate.get("version") != 2
        or cache_gate.get("qualified") is not True
        or cache_gate.get("reasons") != []
        or cache_gate.get("source_revision") != expected_source_revision
        or not isinstance(cache_checks, dict)
        or set(cache_checks) != CACHE_NAMESPACE_CHECKS
        or any(value is not True for value in cache_checks.values())
        or cache_gate.get("error_types") != {}
        or not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef"
                    for character in value)
            for value in cache_module_digests
        )
        or not isinstance(cache_gate.get("pillow_version"), str)
        or not cache_gate.get("pillow_version")
        or not isinstance(cache_privacy, dict)
        or cache_privacy.get("contains_image_bytes") is not False
        or cache_privacy.get("contains_namespace_digest") is not False
        or cache_privacy.get("contains_request_id") is not False
        or cache_privacy.get("contains_prompt_or_output") is not False
        or cache_privacy.get("contains_credentials") is not False
        or cache_gate.get("gpu_execution_required") is not False
        or cache_gate.get("model_execution_performed") is not False
        or cache_gate.get("production_promotion_authorized") is not False
    ):
        reasons.append("M1-89 cache namespace runtime gate did not qualify")

    runtime_identity = diagnostic.get("runtime_identity")
    tool_http_summary = diagnostic.get("tool_http_summary")
    if not isinstance(tool_http_summary, dict):
        tool_http_summary = {}
    if (
        diagnostic.get("schema") != DIAGNOSTIC_SCHEMA
        or diagnostic.get("version") != 1
        or diagnostic.get("qualified") is not True
        or not _exact_zero_gates(
            diagnostic.get("gates"), DIAGNOSTIC_GATE_NAMES)
        or not isinstance(runtime_identity, dict)
        or runtime_identity.get("source_revision") != expected_source_revision
        or runtime_identity.get("physical_gpus") != [str(expected_gpu)]
        or runtime_identity.get("tensor_parallel_size") != 1
        or runtime_identity.get("max_model_len") != 262144
        or tool_http_summary.get(
            "streaming_contract_qualified") is not True
        or tool_http_summary.get(
            "streaming_equivalence_qualified") is not True
        or not _authority_is_diagnostic(diagnostic)
    ):
        reasons.append("M1-84 functional diagnostic did not qualify")
    if diagnostic:
        _check_artifact_manifest(
            diagnostic,
            diagnostic_root,
            "M1-84",
            reasons,
            {name: name for name in DIAGNOSTIC_ARTIFACT_NAMES},
        )

    if (
        image_runner.get("schema") != IMAGE_RUNNER_SCHEMA
        or image_runner.get("version") != 1
        or image_runner.get("qualified") is not True
        or image_runner.get("returncode") != 0
        or image_runner.get("source_revision") != expected_source_revision
        or image_runner.get("physical_gpu") != expected_gpu
        or image_runner.get("terminal_stage") != "completed"
        or not _exact_zero_gates(
            image_runner.get("gates"), IMAGE_RUNNER_GATE_NAMES)
        or not _authority_is_diagnostic(image_runner)
    ):
        reasons.append("M1-86 runner did not qualify")
    decision = image_comparison.get("decision")
    comparison_observed = image_comparison.get("observed")
    if (
        image_comparison.get("schema") != IMAGE_COMPARISON_SCHEMA
        or image_comparison.get("version") != 1
        or image_comparison.get("qualified") is not True
        or not isinstance(decision, dict)
        or not isinstance(comparison_observed, dict)
        or comparison_observed.get("physical_gpu") != expected_gpu
        or decision.get("single_gpu_diagnostic_phase_passed") is not True
        or decision.get("full_model_tp4_required") is not True
        or decision.get("semantic_quality_required") is not True
        or decision.get("production_promotion_authorized") is not False
    ):
        reasons.append("M1-86 multi-image comparison did not qualify")
    if image_runner:
        _check_artifact_manifest(
            image_runner,
            image_root,
            "M1-86 runner",
            reasons,
            IMAGE_RUNNER_ARTIFACTS,
        )
    if image_comparison:
        _check_artifact_manifest(
            image_comparison,
            image_root,
            "M1-86 comparison",
            reasons,
            IMAGE_COMPARISON_ARTIFACTS,
        )

    overlay_trees: list[str] = []
    overlay_sites: list[str] = []
    for label, overlay in (
        ("M1-89", cache_overlay),
        ("M1-84", diagnostic_overlay),
        ("M1-86", image_overlay),
    ):
        tree = overlay.get("runtime_tree_sha256")
        site = overlay.get("runtime_site_packages")
        if (
            overlay.get("schema") != OVERLAY_SCHEMA
            or overlay.get("version") != 1
            or overlay.get("qualified") is not True
            or overlay.get("source_revision") != expected_source_revision
            or not isinstance(tree, str)
            or len(tree) != 64
            or not isinstance(site, str)
            or not Path(site).is_absolute()
        ):
            reasons.append(f"{label} overlay identity did not qualify")
        else:
            overlay_trees.append(tree)
            overlay_sites.append(site)
    if len(overlay_trees) == 3 and len(set(overlay_trees)) != 1:
        reasons.append(
            "M1-89, M1-84, and M1-86 used different runtime overlays")
    if len(overlay_sites) == 3 and len(set(overlay_sites)) != 1:
        reasons.append(
            "M1-89, M1-84, and M1-86 used different runtime paths")
    if (
        cache_gate.get("runtime_site_packages")
        != cache_overlay.get("runtime_site_packages")
    ):
        reasons.append("M1-89 gate and overlay runtime paths differ")

    manifest = (
        runtime_identity.get("diagnostic_manifest_sha256")
        if isinstance(runtime_identity, dict) else None
    )
    model_path = (
        runtime_identity.get("diagnostic_model")
        if isinstance(runtime_identity, dict) else None
    )
    for label, contract in (
        ("control", control_contract),
        ("candidate", candidate_contract),
    ):
        if (
            contract.get("source_revision") != expected_source_revision
            or contract.get("runtime_tree_sha256")
            != (overlay_trees[0] if overlay_trees else None)
            or contract.get("model_manifest_sha256") != manifest
            or contract.get("model_path") != model_path
            or contract.get("tensor_parallel_size") != 1
            or contract.get("max_model_len") != 262144
            or not isinstance(contract.get("environment"), dict)
            or contract.get("environment", {}).get(
                "CUDA_VISIBLE_DEVICES") != str(expected_gpu)
        ):
            reasons.append(f"M1-86 {label} identity differs from M1-84")

    for label, report in (
        ("interstage", interstage_postflight),
        ("final", final_postflight),
    ):
        if (
            report.get("schema") != POSTFLIGHT_SCHEMA
            or report.get("version") != 1
            or report.get("qualified") is not True
            or report.get("gpu_indices") != [expected_gpu]
            or report.get("api_server_pids") != []
            or report.get("worker_pids") != []
            or report.get("gpu_processes") != []
            or report.get("scan_errors") != []
        ):
            reasons.append(f"{label} service postflight did not qualify")
    for label, report in (
        ("interstage", interstage_preflight),
        ("final", final_preflight),
    ):
        if (
            report.get("schema") != "bi100-gpu-preflight-v1"
            or report.get("version") != 1
            or report.get("ok") is not True
            or report.get("gpus") != [expected_gpu]
            or not isinstance(report.get("results"), list)
            or len(report.get("results")) != 1
            or report.get("results", [{}])[0].get("gpu") != expected_gpu
            or report.get("results", [{}])[0].get("ok") is not True
        ):
            reasons.append(f"{label} GPU preflight did not qualify")

    session_identities: list[tuple[int, int, str]] = []
    for label, identity in zip(("M1-84", "M1-86"), child_identities):
        pid = identity.get("pid")
        token = identity.get("session_token")
        if (
            identity.get("schema") != SESSION_SCHEMA
            or identity.get("version") != 1
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 1
            or identity.get("pgid") != pid
            or identity.get("sid") != pid
            or not isinstance(identity.get("starttime_ticks"), int)
            or isinstance(identity.get("starttime_ticks"), bool)
            or identity.get("starttime_ticks", 0) <= 0
            or not isinstance(token, str)
            or len(token) != 32
            or any(character not in "0123456789abcdef"
                   for character in token)
        ):
            reasons.append(f"{label} queue child identity differs")
        else:
            session_identities.append(
                (pid, identity["starttime_ticks"], token))
    if (
        len(session_identities) == 2
        and len(set(session_identities)) != 2
    ):
        reasons.append("queue stages reused one process session")

    actions = recovery.get("actions")
    if (
        recovery.get("schema") != RECOVERY_SCHEMA
        or recovery.get("version") != 1
        or recovery.get("qualified") is not True
        or recovery.get("identity_count") != 5
        or recovery.get("term_grace_s") != 60.0
        or recovery.get("kill_grace_s") != 20.0
        or recovery.get("complete_token_scan_required") is not True
        or not isinstance(actions, list)
        or len(actions) != 5
        or any(
            action.get("initial_live_count") != 0
            or action.get("initial_escaped_count") != 0
            or action.get("token_scan_error_count") != 0
            or action.get("final_live_count") != 0
            or action.get("term_sent") is not False
            or action.get("kill_sent") is not False
            or action.get("outcome") != "already_quiescent"
            for action in actions
            if isinstance(action, dict)
        )
        or any(not isinstance(action, dict) for action in actions or [])
    ):
        reasons.append("recorded service recovery was not clean")

    artifact_paths = {
        "m1_89_cache_namespace_runtime_gate":
            root / "m1_89_cache_namespace_runtime_gate.json",
        "m1_89_overlay":
            root / "m1_89_runtime_overlay_identity.json",
        "m1_84_status": diagnostic_root / "status.json",
        "m1_84_overlay": diagnostic_root / "runtime_overlay_identity.json",
        "m1_86_runner_status": image_root / "runner_status.json",
        "m1_86_comparison": image_root / "comparison.json",
        "m1_86_overlay": image_root / "runtime_overlay_identity.json",
        "interstage_postflight": root / "interstage_postflight.json",
        "interstage_preflight": root / "interstage_preflight.json",
        "final_postflight": root / "final_postflight.json",
        "final_preflight": root / "final_preflight.json",
        "service_recovery": root / "service_recovery.json",
        "m1_84_child_identity": root / "m1_84_child_identity.json",
        "m1_86_child_identity": root / "m1_86_child_identity.json",
    }
    artifact_sha256: dict[str, str] = {}
    for name, path in artifact_paths.items():
        if not _is_confined_regular_file(path, root):
            reasons.append(f"aggregate artifact is not confined: {name}")
            continue
        artifact_sha256[name] = _sha256(path)

    return {
        "schema": SCHEMA,
        "version": 2,
        "qualified": not reasons,
        "reasons": reasons,
        "source_revision": expected_source_revision,
        "physical_gpu": expected_gpu,
        "runner_returncode": runner_returncode,
        "gates": gates,
        "identity": {
            "runtime_tree_sha256":
                overlay_trees[0] if len(set(overlay_trees)) == 1 else None,
            "diagnostic_model": model_path,
            "diagnostic_manifest_sha256": manifest,
        },
        "artifact_sha256": artifact_sha256,
        "decision": {
            "single_gpu_diagnostic_queue_passed": not reasons,
            "full_model_tp4_required": True,
            "official_881_required": True,
            "semantic_quality_required": True,
            "main_or_yaml_change_authorized": False,
            "production_promotion_authorized": False,
        },
        "full_model_evaluated": False,
        "semantic_quality_evaluated": False,
        "performance_evaluated": False,
        "production_promotion_authorized": False,
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-gpu", type=int, required=True)
    parser.add_argument("--runner-returncode", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = qualify(
        args.root,
        expected_source_revision=args.expected_source_revision,
        expected_gpu=args.expected_gpu,
        runner_returncode=args.runner_returncode,
    )
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
