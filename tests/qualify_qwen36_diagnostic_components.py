#!/usr/bin/env python3
"""Fail-closed qualification for Qwen3.6 diagnostic component probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


RELATIVE_L2_LIMIT = 1.0e-5
MOE_FIXED_SPEEDUP_MIN = 1.5
MOE_ROUTED_SPEEDUP_MIN = 1.25
GDN_SPEEDUP_MIN = 1.5
GDN_MEDIAN_MS_MAX = 0.110
GPU_MEMORY_DROP_MAX_BYTES = 1 << 30
PAGED_LENGTHS = {32768, 65536, 131072, 235000}

FATAL_PATTERNS = {
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "segfault": re.compile(r"segmentation fault", re.IGNORECASE),
    "core_dump": re.compile(r"core dumped", re.IGNORECASE),
    "out_of_memory": re.compile(r"out of memory", re.IGNORECASE),
    "illegal_memory": re.compile(
        r"illegal memory access", re.IGNORECASE),
    "cuda_error": re.compile(r"\bCUDA error\b", re.IGNORECASE),
    "gloo_failure": re.compile(
        r"Gloo.*(?:failed|reset)", re.IGNORECASE),
    "worker_loss": re.compile(
        r"worker.*(?:lost|died|terminated unexpectedly)",
        re.IGNORECASE),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _append_limit_reason(
    reasons: list[str],
    value: Any,
    limit: float,
    name: str,
) -> None:
    if not _finite_number(value):
        reasons.append(f"{name} is not finite")
    elif value > limit:
        reasons.append(f"{name} exceeds {limit:g}: {value:g}")


def _check_preflight(
    before: dict[str, Any],
    after: dict[str, Any],
    gpu: int,
    reasons: list[str],
) -> dict[str, Any]:
    if before.get("ok") is not True:
        reasons.append("GPU preflight before probes did not pass")
    if after.get("ok") is not True:
        reasons.append("GPU preflight after probes did not pass")
    if before.get("gpus") != [gpu] or after.get("gpus") != [gpu]:
        reasons.append("GPU preflight physical index differs")

    def free_bytes(report: dict[str, Any]) -> int | None:
        rows = report.get("results")
        if not isinstance(rows, list) or len(rows) != 1:
            return None
        value = rows[0].get("free") if isinstance(rows[0], dict) else None
        return value if isinstance(value, int) and value >= 0 else None

    before_free = free_bytes(before)
    after_free = free_bytes(after)
    drop = (
        max(0, before_free - after_free)
        if before_free is not None and after_free is not None
        else None
    )
    if drop is None:
        reasons.append("GPU free-memory comparison is unavailable")
    elif drop > GPU_MEMORY_DROP_MAX_BYTES:
        reasons.append(
            "GPU free memory dropped by more than 1 GiB after probes")
    return {
        "before_free_bytes": before_free,
        "after_free_bytes": after_free,
        "memory_drop_bytes": drop,
        "memory_drop_limit_bytes": GPU_MEMORY_DROP_MAX_BYTES,
    }


def _check_qgkv(
    reports: list[dict[str, Any]],
    reasons: list[str],
) -> dict[str, Any]:
    ranks: dict[int, dict[str, Any]] = {}
    all_exact = True
    for report in reports:
        rank = report.get("tp_rank")
        if not isinstance(rank, int) or rank in ranks:
            reasons.append("QGKV reports contain an invalid or duplicate rank")
            all_exact = False
            continue
        ranks[rank] = report
    if set(ranks) != {0, 1, 2, 3}:
        reasons.append("QGKV reports do not cover TP4 ranks 0,1,2,3")
        all_exact = False
    for rank, report in ranks.items():
        if report.get("ok") is not True:
            reasons.append(f"QGKV rank {rank} did not pass")
            all_exact = False
        if report.get("loaded") != [True, True, True]:
            reasons.append(f"QGKV rank {rank} did not load q/k/v")
            all_exact = False
        if report.get("weight_exact") != [True, True, True]:
            reasons.append(f"QGKV rank {rank} weight mapping is not exact")
            all_exact = False
        checks = report.get("output_checks")
        if (
            not isinstance(checks, list)
            or len(checks) != 3
            or not all(
                isinstance(row, dict)
                and row.get("exact") is True
                and row.get("max_abs") == 0.0
                for row in checks
            )
        ):
            reasons.append(f"QGKV rank {rank} output is not exact")
            all_exact = False
    return {"tp_ranks": sorted(ranks), "all_exact": all_exact}


def _check_moe(report: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    expected_shape = {
        "experts": 256,
        "top_k": 8,
        "hidden": 2048,
        "intermediate": 128,
        "dtype": "torch.float16",
    }
    if report.get("shape") != expected_shape:
        reasons.append("MoE probe did not use the TP4 rank-local target shape")
    capabilities = report.get("extension_capabilities")
    if not isinstance(capabilities, dict) or not (
        capabilities.get("w13") is True
        and capabilities.get("w2_reduce") is True
    ):
        reasons.append("MoE production staged extension interfaces are missing")
    numerics = report.get("numerics")
    if not isinstance(numerics, dict):
        reasons.append("MoE numerical report is missing")
        numerics = {}
    required_boundaries = ("direct_w13", "direct_w2_reduce", "staged")
    relative_l2: dict[str, Any] = {}
    for name in required_boundaries:
        row = numerics.get(name)
        if not isinstance(row, dict):
            reasons.append(f"MoE {name} numerical row is missing")
            continue
        if row.get("finite") is not True:
            reasons.append(f"MoE {name} produced non-finite output")
        relative_l2[name] = row.get("relative_l2")
        _append_limit_reason(
            reasons,
            row.get("relative_l2"),
            RELATIVE_L2_LIMIT,
            f"MoE {name} relative L2",
        )
    sequence = report.get("sequence", {}).get("staged")
    if not isinstance(sequence, dict):
        reasons.append("MoE staged sequence report is missing")
        sequence = {}
    if sequence.get("finite_steps") != sequence.get("steps"):
        reasons.append("MoE staged sequence contains a non-finite step")
    relative_l2["staged_sequence"] = sequence.get("relative_l2")
    _append_limit_reason(
        reasons,
        sequence.get("relative_l2"),
        RELATIVE_L2_LIMIT,
        "MoE staged sequence relative L2",
    )
    timings = report.get("timings")
    if not isinstance(timings, dict):
        timings = {}
        reasons.append("MoE timing report is missing")
    fixed_speedup = (
        timings.get("staged_fixed", {}).get("speedup_vs_baseline")
        if isinstance(timings.get("staged_fixed"), dict)
        else None
    )
    routed_speedup = (
        timings.get("staged_routed", {}).get("speedup_vs_baseline")
        if isinstance(timings.get("staged_routed"), dict)
        else None
    )
    if not _finite_number(fixed_speedup) or fixed_speedup < MOE_FIXED_SPEEDUP_MIN:
        reasons.append("MoE staged fixed speedup is below 1.5x")
    if (
        not _finite_number(routed_speedup)
        or routed_speedup < MOE_ROUTED_SPEEDUP_MIN
    ):
        reasons.append("MoE staged routed speedup is below 1.25x")
    return {
        "extension_capabilities": capabilities,
        "relative_l2": relative_l2,
        "fixed_speedup": fixed_speedup,
        "routed_speedup": routed_speedup,
    }


def _check_gdn(report: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    config = report.get("config")
    if not isinstance(config, dict) or config.get("shape") != [1, 4, 8, 128]:
        reasons.append("GDN probe did not use the TP4 rank-local target shape")
    sequence = report.get("sequence")
    if not isinstance(sequence, dict):
        sequence = {}
        reasons.append("GDN sequence report is missing")
    if sequence.get("finite_steps") != sequence.get("steps"):
        reasons.append("GDN sequence contains a non-finite step")
    output_l2 = sequence.get("output_relative_l2")
    state_l2 = sequence.get("state_relative_l2")
    _append_limit_reason(
        reasons, output_l2, RELATIVE_L2_LIMIT,
        "GDN output relative L2")
    _append_limit_reason(
        reasons, state_l2, RELATIVE_L2_LIMIT,
        "GDN state relative L2")
    performance = report.get("performance")
    if not isinstance(performance, dict):
        performance = {}
        reasons.append("GDN performance report is missing")
    candidate_ms = performance.get("candidate_median_ms")
    speedup = performance.get("speedup")
    if not _finite_number(candidate_ms) or candidate_ms > GDN_MEDIAN_MS_MAX:
        reasons.append("GDN candidate median exceeds 0.110 ms")
    if not _finite_number(speedup) or speedup < GDN_SPEEDUP_MIN:
        reasons.append("GDN speedup is below 1.5x")
    if report.get("ok") is not True:
        reasons.append("GDN benchmark did not pass its internal gate")
    return {
        "output_relative_l2": output_l2,
        "state_relative_l2": state_l2,
        "candidate_median_ms": candidate_ms,
        "speedup": speedup,
    }


def _check_paged(
    report: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    shape = report.get("shape")
    if not isinstance(shape, dict) or (
        shape.get("head_size") != 256
        or shape.get("block_size") != 16
        or shape.get("dtype") != "float16"
    ):
        reasons.append("paged KV probe tensor contract differs")
    results = report.get("results")
    if not isinstance(results, dict):
        results = {}
        reasons.append("paged KV results are missing")
    parsed_lengths: set[int] = set()
    for key, row in results.items():
        try:
            length = int(key)
        except (TypeError, ValueError):
            reasons.append("paged KV result contains a non-integer length")
            continue
        parsed_lengths.add(length)
        checks = row.get("checks") if isinstance(row, dict) else None
        if not isinstance(checks, dict) or not (
            checks.get("key_exact") is True
            and checks.get("value_exact") is True
            and checks.get("output_exact") is True
            and checks.get("output_max_abs") == 0.0
        ):
            reasons.append(f"paged KV length {length} is not byte-exact")
    missing = PAGED_LENGTHS - parsed_lengths
    if missing:
        reasons.append(f"paged KV required lengths are missing: {sorted(missing)}")
    if report.get("ok") is not True:
        reasons.append("paged KV benchmark did not pass its internal gate")
    return {
        "lengths": sorted(parsed_lengths),
        "all_exact": not missing and report.get("ok") is True,
    }


def _check_cache(
    report: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    gate = report.get("gate")
    if not isinstance(gate, dict) or gate.get("qualified") is not True:
        reasons.append("CacheEngine integration gate did not qualify")
    required = (
        "round_trip_byte_exact",
        "same_slot_preserved_victim_exact",
        "same_slot_promoted_request_exact",
        "invalid_mapping_fail_fast",
        "invalid_mapping_zero_write",
        "invalid_selector_fail_fast",
    )
    failed = [name for name in required if report.get(name) is not True]
    reasons.extend(f"CacheEngine {name} is not true" for name in failed)
    return {
        "qualified": isinstance(gate, dict) and gate.get("qualified") is True,
        "failed_checks": failed,
    }


def _scan_logs(paths: list[Path]) -> tuple[list[dict[str, str]], dict[str, str]]:
    hits: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        hashes[path.name] = _sha256(path)
        for name, pattern in FATAL_PATTERNS.items():
            if pattern.search(text):
                hits.append({"log": path.name, "pattern": name})
    return hits, hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qgkv", type=Path, action="append", required=True)
    parser.add_argument("--moe", type=Path, required=True)
    parser.add_argument("--gdn", type=Path, required=True)
    parser.add_argument("--paged", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--preflight-before", type=Path, required=True)
    parser.add_argument("--preflight-after", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    evidence_paths = (
        args.qgkv
        + [args.moe, args.gdn, args.paged, args.cache,
           args.preflight_before, args.preflight_after,
           args.runtime_identity]
    )
    reasons: list[str] = []
    report: dict[str, Any] = {
        "schema": "qwen36-diagnostic-component-gate-v1",
        "version": 1,
        "scope": "single-GPU structural and TP4-rank-local component probes",
        "source_revision": args.source_revision,
        "source_branch": args.source_branch,
        "instance": args.instance,
        "physical_gpu": args.physical_gpu,
        "thresholds": {
            "relative_l2_max": RELATIVE_L2_LIMIT,
            "moe_fixed_speedup_min": MOE_FIXED_SPEEDUP_MIN,
            "moe_routed_speedup_min": MOE_ROUTED_SPEEDUP_MIN,
            "gdn_speedup_min": GDN_SPEEDUP_MIN,
            "gdn_candidate_median_ms_max": GDN_MEDIAN_MS_MAX,
        },
        "semantic_quality_evaluated": False,
        "full_model_tp4_evaluated": False,
        "production_promotion_authorized": False,
    }
    try:
        qgkv = [_load(path) for path in args.qgkv]
        moe = _load(args.moe)
        gdn = _load(args.gdn)
        paged = _load(args.paged)
        cache = _load(args.cache)
        before = _load(args.preflight_before)
        after = _load(args.preflight_after)
        runtime = _load(args.runtime_identity)
        if runtime.get("qualified") is not True:
            reasons.append("immutable runtime identity did not qualify")
        if runtime.get("source_revision") != args.source_revision:
            reasons.append("runtime source revision differs")
        report["runtime"] = {
            "qualified": runtime.get("qualified"),
            "runtime_tree_sha256": runtime.get("runtime_tree_sha256"),
        }
        report["gpu_health"] = _check_preflight(
            before, after, args.physical_gpu, reasons)
        report["qgkv"] = _check_qgkv(qgkv, reasons)
        report["moe"] = _check_moe(moe, reasons)
        report["gdn"] = _check_gdn(gdn, reasons)
        report["paged_kv"] = _check_paged(paged, reasons)
        report["cache_engine"] = _check_cache(cache, reasons)
        fatal_hits, log_hashes = _scan_logs(args.log)
        if fatal_hits:
            reasons.append("fatal pattern found in component logs")
        report["fatal_scan"] = {
            "hits": fatal_hits,
            "log_sha256": log_hashes,
        }
        report["evidence_sha256"] = {
            path.name: _sha256(path) for path in evidence_paths
        }
    except Exception as exc:
        reasons.append(f"qualification error {type(exc).__name__}: {exc}")

    report["qualified"] = not reasons
    report["reasons"] = reasons
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out),
        "qualified": report["qualified"],
        "reasons": reasons,
    }, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
