#!/usr/bin/env python3
"""Qualify M1-103 execution without treating candidate rejection as run failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


Json = dict[str, Any]
SCHEMA = "bi100-m1-103-legacy-oracle-queue-v1"
PREFIX_SCHEMA = "bi100-m1-100-prefix-cold-high-precision-v1"
WMMA_SCHEMA = "bi100-m1-101-wmma-qk-high-precision-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PREFIX_FROZEN = {
    "bench_prefix_attention_breakdown.py":
        "2ab82f69e7833dc2965b03e4cbcebe5beafd9d4954a3e3babda101bb54a0ddd2",
    "bench_prefix_cold_chunk_hybrid.py":
        "e2dffa151c99f4cf28d827877db68bbcb0a0c0bd6433c466017c255df2f3d076",
}
WMMA_FROZEN = {
    "bench_attention_wmma_qk.py":
        "55a4ed735abda6e88f2bbb3f4cc264af1b9629062fb62c9dfc130f683c63895f",
    "build_corex_attention_wmma_qk_probe.sh":
        "9436cd30428f357addf3bcf90d14618a984d48d08f593ac88db70dc6da688958",
    "corex_attention_wmma_qk_probe.cu":
        "08a68ffc068c7f5a21796b32b64e2164c03f7c1b0270e19d862e116abdd3c688",
}
ZERO_GATES = (
    "postflight_before",
    "preflight_before",
    "wmma_build",
    "child_cleanup",
    "recovery",
    "recovery_clean",
    "postflight",
    "preflight_after",
    "preflight_comparison",
    "fatal_scan",
    "timeout_scan",
)


def _load(path: Path) -> Json:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_rc(path: Path) -> int | None:
    try:
        value = _read_text(path)
    except OSError:
        return None
    return int(value) if value.isdigit() else None


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _validate_common_report(
    value: Json,
    *,
    schema: str,
    frozen: dict[str, str],
    reasons: list[str],
    label: str,
) -> tuple[bool | None, Json]:
    summary = value.get("summary")
    decision = summary.get("decision") if isinstance(summary, dict) else None
    if (
        value.get("schema") != schema
        or value.get("version") != 1
        or value.get("frozen_artifacts") != frozen
        or not isinstance(summary, dict)
        or not isinstance(summary.get("qualified"), bool)
        or not isinstance(summary.get("reasons"), list)
        or not isinstance(decision, dict)
    ):
        reasons.append(f"{label} report contract differs")
        return None, {}
    for forbidden in (
        "service_integration_authorized",
        "production_promotion_authorized",
        "yaml_change_authorized",
        "main_merge_authorized",
    ):
        if decision.get(forbidden) is not False:
            reasons.append(f"{label} unexpectedly authorizes {forbidden}")
    return summary["qualified"], decision


def _validate_identity(
    value: Json,
    *,
    label: str,
    reasons: list[str],
) -> int | None:
    pid = value.get("pid")
    token = value.get("session_token")
    if (
        value.get("schema") != "bi100-process-session-v1"
        or value.get("version") != 1
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or value.get("pgid") != pid
        or value.get("sid") != pid
        or not isinstance(value.get("starttime_ticks"), int)
        or value.get("starttime_ticks") <= 0
        or not isinstance(token, str)
        or len(token) != 32
        or any(character not in "0123456789abcdef"
               for character in token)
    ):
        reasons.append(f"{label} process identity differs")
        return None
    return pid


def qualify(
    root: Path,
    *,
    expected_source_revision: str,
    expected_prefix_gpu: int,
    expected_wmma_gpu: int,
    runner_returncode: int,
) -> Json:
    reasons: list[str] = []
    gates = {
        name: _read_rc(root / f"{name}.rc")
        for name in ZERO_GATES
    }
    candidate_rcs = {
        "prefix": _read_rc(root / "prefix.rc"),
        "wmma": _read_rc(root / "wmma.rc"),
    }
    if any(value != 0 for value in gates.values()):
        reasons.append("one or more infrastructure/lifecycle gates failed")
    if any(value not in (0, 1) for value in candidate_rcs.values()):
        reasons.append("candidate return codes must be 0 or 1")
    if runner_returncode != 0:
        reasons.append("runner return code is nonzero")

    try:
        source_revision = _read_text(root / "source_revision.txt")
        source_branch = _read_text(root / "source_branch.txt")
        instance = _read_text(root / "instance.txt")
        prefix_gpu = int(_read_text(root / "prefix_gpu.txt"))
        wmma_gpu = int(_read_text(root / "wmma_gpu.txt"))
        stage = _read_text(root / "stage.txt")
    except (OSError, ValueError):
        source_revision = None
        source_branch = None
        instance = None
        prefix_gpu = None
        wmma_gpu = None
        stage = None
        reasons.append("runner identity files are incomplete")
    if source_revision != expected_source_revision:
        reasons.append("source revision differs")
    if not isinstance(source_branch, str) or not source_branch:
        reasons.append("source branch is missing")
    if not isinstance(instance, str) or not instance:
        reasons.append("instance label is missing")
    if prefix_gpu != expected_prefix_gpu or wmma_gpu != expected_wmma_gpu:
        reasons.append("physical GPU assignment differs")
    if prefix_gpu == wmma_gpu:
        reasons.append("oracle GPU assignments are not distinct")
    if stage != "completed":
        reasons.append("runner did not reach completed stage")

    prefix_qualified = None
    wmma_qualified = None
    prefix_decision: Json = {}
    wmma_decision: Json = {}
    try:
        prefix = _load(root / "prefix" / "report.json")
    except (OSError, ValueError, json.JSONDecodeError):
        prefix = {}
        reasons.append("prefix report is invalid")
    else:
        prefix_qualified, prefix_decision = _validate_common_report(
            prefix,
            schema=PREFIX_SCHEMA,
            frozen=PREFIX_FROZEN,
            reasons=reasons,
            label="prefix",
        )
        if (
            prefix.get("config", {}).get("production_query_len") != 8176
            or prefix.get("config", {}).get("primary_context") != 65536
            or prefix.get("config", {}).get("partial_context") != 65552
            or prefix.get("config", {}).get(
                "minimum_primary_reduction") != 0.15
        ):
            reasons.append("prefix fixed configuration differs")
        if prefix_decision.get(
                "next_token_gate_authorized") is not prefix_qualified:
            reasons.append("prefix decision does not match qualification")

    try:
        wmma = _load(root / "wmma" / "report.json")
    except (OSError, ValueError, json.JSONDecodeError):
        wmma = {}
        reasons.append("WMMA report is invalid")
    else:
        wmma_qualified, wmma_decision = _validate_common_report(
            wmma,
            schema=WMMA_SCHEMA,
            frozen=WMMA_FROZEN,
            reasons=reasons,
            label="WMMA",
        )
        if (
            wmma.get("config", {}).get("tiles") != 128
            or wmma.get("config", {}).get("head_dim") != 256
            or wmma.get("config", {}).get("minimum_qk_speedup") != 1.5
            or not SHA256_RE.fullmatch(
                str(wmma.get("extension_sha256") or ""))
        ):
            reasons.append("WMMA fixed configuration differs")
        if wmma_decision.get(
                "integration_benefit_gate_authorized") is not wmma_qualified:
            reasons.append("WMMA decision does not match qualification")

    for label, qualified in (
        ("prefix", prefix_qualified),
        ("wmma", wmma_qualified),
    ):
        expected_rc = 0 if qualified is True else 1
        if candidate_rcs[label] != expected_rc:
            reasons.append(f"{label} return code does not match report")

    for label, relative in (
        ("pre-run postflight", "postflight_before.json"),
        ("recovery clean", "recovery_clean.json"),
        ("postflight", "postflight.json"),
        ("preflight comparison", "preflight_comparison.json"),
    ):
        try:
            value = _load(root / relative)
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append(f"{label} report is invalid")
        else:
            if value.get("qualified") is not True:
                reasons.append(f"{label} is not qualified")
    try:
        recovery_clean = _load(root / "recovery_clean.json")
    except (OSError, ValueError, json.JSONDecodeError):
        recovery_clean = {}
    if recovery_clean.get("emergency_recovery_used") is not False:
        reasons.append("recorded child required emergency recovery")

    identity_pids = []
    for label in ("prefix", "wmma"):
        try:
            identity = _load(root / f"{label}_identity.json")
        except (OSError, ValueError, json.JSONDecodeError):
            reasons.append(f"{label} process identity is invalid")
        else:
            identity_pids.append(_validate_identity(
                identity, label=label, reasons=reasons))
    if len(identity_pids) != 2 or None in identity_pids \
            or len(set(identity_pids)) != 2:
        reasons.append("process identities are not two unique sessions")

    for path in (root / "fatal_scan.txt", root / "timeout_scan.txt"):
        try:
            if path.read_text(encoding="utf-8"):
                reasons.append(f"{path.name} is not empty")
        except OSError:
            reasons.append(f"{path.name} is missing")

    artifacts = {
        name: _sha256(root / relative)
        for name, relative in (
            ("prefix_report", "prefix/report.json"),
            ("wmma_report", "wmma/report.json"),
            ("prefix_identity", "prefix_identity.json"),
            ("wmma_identity", "wmma_identity.json"),
            ("postflight_before", "postflight_before.json"),
            ("recovery_clean", "recovery_clean.json"),
            ("postflight", "postflight.json"),
            ("preflight_comparison", "preflight_comparison.json"),
        )
    }
    if any(value is None for value in artifacts.values()):
        reasons.append("one or more bound artifacts are missing")

    return {
        "schema": SCHEMA,
        "version": 1,
        "qualified": not reasons,
        "reasons": reasons,
        "source_revision": source_revision,
        "source_branch": source_branch,
        "instance": instance,
        "physical_gpus": {
            "prefix": prefix_gpu,
            "wmma": wmma_gpu,
        },
        "runner_returncode": runner_returncode,
        "gates": gates,
        "candidate_returncodes": candidate_rcs,
        "candidates": {
            "prefix": {
                "qualified": prefix_qualified,
                "next_token_gate_authorized": (
                    prefix_decision.get("next_token_gate_authorized")
                ),
            },
            "wmma": {
                "qualified": wmma_qualified,
                "integration_benefit_gate_authorized": (
                    wmma_decision.get(
                        "integration_benefit_gate_authorized")
                ),
            },
        },
        "artifacts": artifacts,
        "production_promotion_authorized": False,
        "yaml_change_authorized": False,
        "main_merge_authorized": False,
        "privacy": {
            "contains_raw_tensors": False,
            "contains_model_output": False,
            "contains_session_tokens": False,
            "contains_credentials": False,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-prefix-gpu", type=int, required=True)
    parser.add_argument("--expected-wmma-gpu", type=int, required=True)
    parser.add_argument("--runner-returncode", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = qualify(
        args.root,
        expected_source_revision=args.expected_source_revision,
        expected_prefix_gpu=args.expected_prefix_gpu,
        expected_wmma_gpu=args.expected_wmma_gpu,
        runner_returncode=args.runner_returncode,
    )
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
