#!/usr/bin/env python3
"""Build a privacy-safe qualification record for the M1-48 path profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

try:
    from summarize_prefill_path_profile import summarize as summarize_profile
except ImportError:
    from tests.summarize_prefill_path_profile import (
        summarize as summarize_profile,
    )


SCHEMA = "bi100-m1-48-prefill-profile-qualification-v1"
VERSION = 1
M1_49_SCHEMA = "bi100-m1-49-long-context-qualification-v1"
STARTUP_SCHEMA = "bi100-hybrid-kv-startup-v1"
PREFLIGHT_SCHEMA = "bi100-gpu-preflight-comparison-v1"
SERVICE_SCHEMA = "bi100-m1-48-prefill-service-v1"
PROFILE_SCHEMA = "bi100-m1-48-prefill-path-profile-v2"
RUNTIME_SCHEMA = "bi100-bare-host-runtime-install-v2"
RUNTIME_IDENTITY_SCHEMA = "bi100-m1-48-runtime-identity-v1"
MAX_FREE_MEMORY_DROP_BYTES = 1_073_741_824
MAX_MODEL_RANK_SPREAD_FRACTION = 0.10
PROFILE_FILTER = (
    "model.*,layer.*,full_attn.*,xformers.*,paged_attn.*,moe.*,"
    "gdn_prefix.*"
)
FATAL_RE = re.compile(
    r"CUDA error|SIGSEGV|Fatal Python error|out of memory|"
    r"worker process.*died|Gloo.*failed|AssertionError",
    re.IGNORECASE,
)
STARTUP_INVARIANTS = (
    "config_mode",
    "expected_attention_layers",
    "observed_attention_layers",
    "observed_layer_count",
    "full_attention_ordinals",
    "num_key_value_heads",
    "rank_kv_heads",
    "head_dim",
    "tensor_parallel_size",
    "dtype",
    "dtype_bytes",
    "expected_kv_bytes_per_block",
    "max_model_len_required",
    "block_size",
    "required_gpu_blocks",
    "observed_max_seq_len",
    "runtime_contract",
    "runtime_contract_invariant_sha256",
)


def _valid_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_m1_49(value: Any, reasons: list[str]) -> None:
    if (not isinstance(value, dict)
            or value.get("schema") != M1_49_SCHEMA
            or value.get("version") != 1
            or value.get("qualified") is not True
            or value.get("reasons") != []):
        reasons.append("M1-49 long-context prerequisite is not qualified")
        return
    if value.get("scope") != "hybrid-kv-capacity-correctness-not-prefill-speed":
        reasons.append("M1-49 prerequisite scope is invalid")
    candidate = value.get("candidate_startup") or {}
    if (candidate.get("mode") != "full_attention"
            or candidate.get("attention_layers") != 10):
        reasons.append("M1-49 prerequisite is not the ten-layer candidate")


def _validate_runtime(value: Any, reasons: list[str]) -> None:
    if (not isinstance(value, dict)
            or value.get("schema") != RUNTIME_SCHEMA
            or value.get("version") != 2
            or value.get("qualified") is not True
            or value.get("system_site_packages_modified") is not False
            or value.get("startup_profile_guard_patch") is not True
            or not _valid_digest(value.get("worker_sha256"))):
        reasons.append("M1-48 atomic runtime install is not qualified")
        return
    files = value.get("files")
    required = {"vllm_model", "bi100_profile", "paged_attention",
                "xformers_backend"}
    if not isinstance(files, dict) or not required.issubset(files):
        reasons.append("M1-48 runtime install is missing profiled files")
        return
    for name in sorted(required):
        record = files.get(name) or {}
        if (record.get("same") is not True
                or not _valid_digest(record.get("source_sha256"))
                or record.get("source_sha256")
                != record.get("installed_sha256")):
            reasons.append(f"M1-48 runtime file identity failed: {name}")


def _validate_runtime_identity(
    value: Any,
    source_revision: str,
    reasons: list[str],
) -> None:
    if (not isinstance(value, dict)
            or value.get("schema") != RUNTIME_IDENTITY_SCHEMA
            or value.get("version") != 1
            or value.get("qualified") is not True
            or value.get("reasons") != []):
        reasons.append("M1-48 current/runtime identity is not qualified")
        return
    if value.get("source_revision") != source_revision.strip():
        reasons.append("M1-48 runtime identity source revision differs")
    if (value.get("startup_profile_guard_patch") is not True
            or not _valid_digest(value.get("install_worker_sha256"))
            or value.get("install_worker_sha256")
            != value.get("runtime_worker_sha256")):
        reasons.append("M1-48 active runtime startup-profile guard differs")
    files = value.get("files")
    expected = {"vllm_model", "bi100_profile", "paged_attention",
                "xformers_backend"}
    if not isinstance(files, dict) or set(files) != expected:
        reasons.append("M1-48 runtime identity file set is invalid")
        return
    for name in sorted(expected):
        record = files[name]
        hashes = {
            record.get("current_source_sha256"),
            record.get("install_source_sha256"),
            record.get("installed_sha256"),
            record.get("runtime_installed_sha256"),
        }
        if (record.get("same") is not True or len(hashes) != 1
                or not _valid_digest(next(iter(hashes)))):
            reasons.append(f"M1-48 runtime identity failed: {name}")


def _validate_startups(
    control: Any,
    profile: Any,
    reasons: list[str],
) -> None:
    values = {"control": control, "profile": profile}
    for name, value in values.items():
        if (not isinstance(value, dict)
                or value.get("schema") != STARTUP_SCHEMA
                or value.get("version") != 1
                or value.get("qualified") is not True
                or value.get("reasons") != []):
            reasons.append(f"{name} startup gate is not qualified")
            continue
        if (value.get("mode") != "full_attention"
                or value.get("observed_attention_layers") != 10
                or value.get("tensor_parallel_size") != 4
                or value.get("max_model_len_required") != 262144
                or value.get("block_size") != 16):
            reasons.append(f"{name} startup contract differs from M1-48")
    if isinstance(control, dict) and isinstance(profile, dict):
        for field in STARTUP_INVARIANTS:
            if control.get(field) != profile.get(field):
                reasons.append(
                    f"control/profile startup differ in {field}")


def _validate_preflight(value: Any, reasons: list[str]) -> None:
    if (not isinstance(value, dict)
            or value.get("schema") != PREFLIGHT_SCHEMA
            or value.get("version") != 1
            or value.get("qualified") is not True
            or value.get("reasons") != []):
        reasons.append("M1-48 GPU preflight comparison is not qualified")
        return
    if value.get("max_free_memory_drop_bytes") != MAX_FREE_MEMORY_DROP_BYTES:
        reasons.append("M1-48 GPU memory-drop bound differs from 1 GiB")
    stages = value.get("stages")
    expected_labels = ["before_control", "after_control", "after_profile"]
    if (not isinstance(stages, list)
            or [stage.get("label") for stage in stages
                if isinstance(stage, dict)] != expected_labels):
        reasons.append("M1-48 preflight stage order is invalid")
        return
    for stage in stages:
        results = stage.get("results") if isinstance(stage, dict) else None
        if (stage.get("qualified") is not True
                or not isinstance(results, list)
                or [result.get("gpu") for result in results
                    if isinstance(result, dict)] != [0, 1, 2, 3]
                or any(result.get("ok") is not True for result in results
                       if isinstance(result, dict))):
            reasons.append(
                f"invalid GPU preflight stage: {stage.get('label')}")


def _validate_service(
    value: Any,
    mode: str,
    reasons: list[str],
) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or value.get("schema") != SERVICE_SCHEMA
            or value.get("version") != 1
            or value.get("mode") != mode
            or value.get("qualified_measurement") is not True
            or value.get("reasons") != []):
        reasons.append(f"{mode} service measurement is invalid")
        return {}
    protocol = value.get("protocol") or {}
    expected = {
        "stream": True,
        "max_tokens": 1,
        "min_tokens": 1,
        "temperature": 0,
        "seed": 20260722,
        "thinking": False,
        "target_prompt_tokens": 235000,
        "max_model_len": 262144,
    }
    if protocol != expected:
        reasons.append(f"{mode} service protocol is invalid")
    request = value.get("request") or {}
    if (request.get("prompt_tokens") != 235000
            or request.get("cached_tokens") != 0
            or request.get("completion_tokens") != 1
            or not _finite(request.get("ttft_s"))
            or request.get("ttft_s", 0) <= 0
            or not _valid_digest(request.get("output_sha256"))):
        reasons.append(f"{mode} service request is invalid")
    return request


def _validate_profile(
    value: Any,
    recomputed: Any,
    control_request: dict[str, Any],
    profile_request: dict[str, Any],
    reasons: list[str],
) -> None:
    if _canonical_json(value) != _canonical_json(recomputed):
        reasons.append("M1-48 profile summary differs from source recomputation")
    if (not isinstance(value, dict)
            or value.get("schema") != PROFILE_SCHEMA
            or value.get("version") != 2
            or value.get("qualified_profile") is not True
            or value.get("reasons") != []):
        reasons.append("M1-48 profile summary is not qualified")
        return
    request = value.get("request") or {}
    if (request.get("prefill_tokens") != 235000
            or request.get("expected_chunk_size") != 8192
            or request.get("block_size") != 16
            or request.get("group_index") != 0
            or request.get("forward_count") != 29
            or request.get("tp_ranks") != [0, 1, 2, 3]
            or request.get("profile_overhead_limit_fraction") != 0.15):
        reasons.append("M1-48 profile request contract is invalid")
    overhead = request.get("profile_overhead_fraction")
    if not _finite(overhead) or abs(overhead) > 0.15:
        reasons.append("M1-48 profile overhead is invalid")
    control_ttft = control_request.get("ttft_s")
    profile_ttft = profile_request.get("ttft_s")
    if (not _finite(control_ttft) or not _finite(profile_ttft)
            or control_ttft <= 0 or profile_ttft <= 0
            or request.get("control_ttft_s") != control_ttft
            or request.get("profile_ttft_s") != profile_ttft):
        reasons.append("M1-48 profile TTFT is not bound to service evidence")
    elif _finite(overhead) and not math.isclose(
            overhead,
            profile_ttft / control_ttft - 1.0,
            rel_tol=1e-12,
            abs_tol=1e-12):
        reasons.append("M1-48 profile overhead does not match service TTFT")
    if request.get("control_output_sha256") != control_request.get(
            "output_sha256"):
        reasons.append("M1-48 profile output digest differs from service")
    for field in (
            "model_rank_spread_fraction",
            "max_forward_model_rank_spread_fraction"):
        spread = request.get(field)
        if (not _finite(spread) or spread < 0
                or spread > MAX_MODEL_RANK_SPREAD_FRACTION):
            reasons.append(f"M1-48 profile {field} is invalid")
    full_attention = value.get("full_attention")
    required_full_attention = {
        "inclusive_ms_per_rank_mean",
        "paged_segment_ms_per_rank_mean",
        "attention_unattributed_ms_per_rank_mean",
        "paged_unattributed_ms_per_rank_mean",
    }
    if (not isinstance(full_attention, dict)
            or not required_full_attention.issubset(full_attention)):
        reasons.append("M1-48 full-attention profile is missing")
    else:
        for field in sorted(required_full_attention):
            value_at_field = full_attention.get(field)
            if (not _finite(value_at_field)
                    or (field in {
                        "inclusive_ms_per_rank_mean",
                        "paged_segment_ms_per_rank_mean",
                    } and value_at_field <= 0)):
                reasons.append(
                    f"M1-48 full-attention field is invalid: {field}")


def _validate_cleanup(value: str, reasons: list[str]) -> None:
    if value.strip() != "0":
        reasons.append("M1-48 pre-qualification cleanup did not pass")


def _validate_logs(
    control_log: str,
    profile_log: str,
    reasons: list[str],
) -> None:
    common = [
        "[BI100] runtime overlay active:",
        "[BI100] GDN cache; policy=admission64 restore=direct",
        "[BI100] M1-49 runtime contract; accounting=full_attention",
    ]
    for name, text in (("control", control_log), ("profile", profile_log)):
        for marker in common:
            if marker not in text:
                reasons.append(f"{name} log is missing {marker}")
        expected_profile = (
            "[BI100] M1-48 profile contract; "
            f"enabled={'1' if name == 'profile' else '0'} mode=event "
            f"include_startup=0 filter={PROFILE_FILTER}"
        )
        if expected_profile not in text:
            reasons.append(f"{name} log has the wrong profile contract")
        if FATAL_RE.search(text):
            reasons.append(f"{name} log contains a fatal signature")
    if "[BI100_PROFILE_EVENT]" in control_log:
        reasons.append("control log unexpectedly contains profile events")
    if "[BI100_PROFILE_EVENT]" not in profile_log:
        reasons.append("profile log does not contain profile events")


def qualify(
    *,
    m1_49: Any,
    runtime_install: Any,
    runtime_identity: Any,
    preflight: Any,
    control_startup: Any,
    profile_startup: Any,
    control_service: Any,
    profile_service: Any,
    profile_summary: Any,
    recomputed_profile_summary: Any,
    prequalification_cleanup: str,
    source_revision: str,
    control_log: str,
    profile_log: str,
    source_sha256: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    _validate_m1_49(m1_49, reasons)
    _validate_runtime(runtime_install, reasons)
    _validate_runtime_identity(runtime_identity, source_revision, reasons)
    _validate_preflight(preflight, reasons)
    _validate_startups(control_startup, profile_startup, reasons)
    control_request = _validate_service(control_service, "control", reasons)
    profile_request = _validate_service(profile_service, "profile", reasons)
    _validate_profile(
        profile_summary,
        recomputed_profile_summary,
        control_request,
        profile_request,
        reasons,
    )
    _validate_cleanup(prequalification_cleanup, reasons)
    _validate_logs(control_log, profile_log, reasons)
    normalized_revision = source_revision.strip()
    if re.fullmatch(r"[0-9a-f]{40}", normalized_revision) is None:
        reasons.append("M1-48 source revision is invalid")
    if (isinstance(control_service, dict) and isinstance(profile_service, dict)
            and control_service.get("run_id") != profile_service.get("run_id")):
        reasons.append("control/profile service run IDs differ")
    if (control_request.get("output_sha256")
            != profile_request.get("output_sha256")):
        reasons.append("control/profile output digests differ")

    expected_sources = {
        "m1_49",
        "runtime_install",
        "runtime_identity",
        "preflight",
        "control_startup",
        "profile_startup",
        "control_service",
        "profile_service",
        "profile_summary",
        "prequalification_cleanup",
        "source_revision",
        "control_log",
        "profile_log",
    }
    if set(source_sha256) != expected_sources:
        reasons.append("M1-48 source digest set is invalid")
    for name in sorted(expected_sources):
        if not _valid_digest(source_sha256.get(name)):
            reasons.append(f"M1-48 source digest is invalid: {name}")

    request = profile_summary.get("request", {}) \
        if isinstance(profile_summary, dict) else {}
    full_attention = profile_summary.get("full_attention", {}) \
        if isinstance(profile_summary, dict) else {}
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "scope": "post-m1-49-diagnostic-path-ranking-only",
        "promotion_authorized": False,
        "source_revision": normalized_revision,
        "request": {
            "prefill_tokens": request.get("prefill_tokens"),
            "forward_count": request.get("forward_count"),
            "tp_ranks": request.get("tp_ranks"),
            "control_ttft_s": request.get("control_ttft_s"),
            "profile_ttft_s": request.get("profile_ttft_s"),
            "profile_overhead_fraction": request.get(
                "profile_overhead_fraction"),
            "model_rank_spread_fraction": request.get(
                "model_rank_spread_fraction"),
            "max_forward_model_rank_spread_fraction": request.get(
                "max_forward_model_rank_spread_fraction"),
        },
        "full_attention": {
            "inclusive_ms_per_rank_mean": full_attention.get(
                "inclusive_ms_per_rank_mean"),
            "paged_segment_ms_per_rank_mean": full_attention.get(
                "paged_segment_ms_per_rank_mean"),
            "attention_unattributed_ms_per_rank_mean": full_attention.get(
                "attention_unattributed_ms_per_rank_mean"),
            "paged_unattributed_ms_per_rank_mean": full_attention.get(
                "paged_unattributed_ms_per_rank_mean"),
        },
        "source_sha256": source_sha256,
    }


def _load(path: Path) -> tuple[Any, str]:
    payload = path.read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def _load_text(path: Path) -> tuple[str, str]:
    payload = path.read_bytes()
    return payload.decode("utf-8", "replace"), hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-49", type=Path, required=True)
    parser.add_argument("--runtime-install", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--control-startup", type=Path, required=True)
    parser.add_argument("--profile-startup", type=Path, required=True)
    parser.add_argument("--control-service", type=Path, required=True)
    parser.add_argument("--profile-service", type=Path, required=True)
    parser.add_argument("--profile-summary", type=Path, required=True)
    parser.add_argument(
        "--prequalification-cleanup", type=Path, required=True)
    parser.add_argument("--source-revision", type=Path, required=True)
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--profile-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        sources = {
            "m1_49": _load(args.m1_49),
            "runtime_install": _load(args.runtime_install),
            "runtime_identity": _load(args.runtime_identity),
            "preflight": _load(args.preflight),
            "control_startup": _load(args.control_startup),
            "profile_startup": _load(args.profile_startup),
            "control_service": _load(args.control_service),
            "profile_service": _load(args.profile_service),
            "profile_summary": _load(args.profile_summary),
            "prequalification_cleanup": _load_text(
                args.prequalification_cleanup),
            "source_revision": _load_text(args.source_revision),
            "control_log": _load_text(args.control_log),
            "profile_log": _load_text(args.profile_log),
        }
        report = qualify(
            m1_49=sources["m1_49"][0],
            runtime_install=sources["runtime_install"][0],
            runtime_identity=sources["runtime_identity"][0],
            preflight=sources["preflight"][0],
            control_startup=sources["control_startup"][0],
            profile_startup=sources["profile_startup"][0],
            control_service=sources["control_service"][0],
            profile_service=sources["profile_service"][0],
            profile_summary=sources["profile_summary"][0],
            recomputed_profile_summary=summarize_profile(
                args.profile_log,
                expected_prefill_tokens=235000,
                expected_processes=4,
                profile_service=args.profile_service,
                control_service=args.control_service,
                expected_chunk_size=8192,
                block_size=16,
            ),
            prequalification_cleanup=(
                sources["prequalification_cleanup"][0]),
            source_revision=sources["source_revision"][0],
            control_log=sources["control_log"][0],
            profile_log=sources["profile_log"][0],
            source_sha256={name: value[1]
                           for name, value in sources.items()},
        )
    except Exception as error:
        report = {
            "schema": SCHEMA,
            "version": VERSION,
            "qualified": False,
            "reasons": [f"cannot qualify M1-48 evidence: {error}"],
            "scope": "post-m1-49-diagnostic-path-ranking-only",
            "promotion_authorized": False,
        }
    _atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
