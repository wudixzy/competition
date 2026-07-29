#!/usr/bin/env python3
"""Finalize M1-137 capability evidence after scoped TP4 cleanup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable


TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
import compare_m1_137_ifeval_power_ab as m1_137


AGGREGATE_SCHEMA = "bi100-m1-137-ifeval-power149-fused-prefill-ab-v1"
SCHEMA = "bi100-m1-137-ifeval-power149-final-qualification-v1"
VERSION = 1
EXPECTED_RCS = (
    "control",
    "candidate",
    "ifeval_paired_noninferiority",
    "aggregate",
    "orchestrator_cleanup",
    "orchestrator_recovery",
    "orchestrator_recovery_clean",
    "orchestrator_postflight",
    "orchestrator_preflight_after",
    "orchestrator_fatal_scan",
    "orchestrator_timeout_scan",
)
EVIDENCE_FILES = (
    "aggregate.json",
    "orchestrator_recovery.json",
    "orchestrator_recovery_clean.json",
    "orchestrator_postflight.json",
    "orchestrator_preflight_after.json",
    "orchestrator_fatal_scan.txt",
    "orchestrator_timeout_scan.txt",
)
Json = dict[str, Any]
FATAL_PATTERN = re.compile(
    rb"CUDA error|illegal memory access|SIGSEGV|Fatal Python error|"
    rb"out of memory|device-side assert|AssertionError|"
    rb"CoreX.*(failed|fatal)|Gloo.*(failed|reset|error)|"
    rb"NCCL.*(failed|abort|error)|Connection reset by peer|"
    rb"worker.*(died|lost|exited unexpectedly)|"
    rb"Timeout(Error|Expired)|engine iteration timed out|"
    rb"watchdog.*tim(e|ed) out|"
    rb"scheduler requested a missing GDN prefix state|"
    rb"non-finite GatedDeltaNet",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rc(root: Path, name: str, reasons: list[str]) -> int | None:
    path = root / f"{name}.rc"
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        reasons.append(f"{name} return code is missing")
        return None
    if not raw.isdigit():
        reasons.append(f"{name} return code is malformed")
        return None
    value = int(raw)
    if value != 0:
        reasons.append(f"{name} return code is {value}")
    return value


def _load_json(
    path: Path,
    label: str,
    reasons: list[str],
) -> Json | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        reasons.append(f"{label} report is missing or malformed")
        return None
    if not isinstance(value, dict):
        reasons.append(f"{label} report root must be an object")
        return None
    return value


def _aggregate_reasons(value: Json | None) -> list[str]:
    if value is None:
        return []
    if (
        value.get("schema") != AGGREGATE_SCHEMA
        or value.get("version") != 1
        or value.get("qualified") is not True
        or value.get(
            "ifeval_two_point_capability_surface_statistically_qualified"
        ) is not True
        or value.get(
            "ifeval_two_point_capability_surface_authorized"
        ) is not False
        or value.get("outer_lifecycle_pending") is not True
        or value.get("reasons") != []
    ):
        return ["pre-cleanup M1-137 aggregate did not qualify"]
    for name in (
        "performance_authorized",
        "default_change_authorized",
        "yaml_change_authorized",
        "main_merge_authorized",
        "production_promotion_authorized",
    ):
        if value.get(name) is not False:
            return ["pre-cleanup M1-137 authorization boundary differs"]
    if value.get("privacy") != {
        "contains_raw_requests": False,
        "contains_raw_model_outputs": False,
        "contains_sample_outcomes": False,
        "contains_credentials": False,
    }:
        return ["pre-cleanup M1-137 privacy contract differs"]
    return []


def _recovery_reasons(
    recovery: Json | None,
    clean: Json | None,
    recovery_path: Path,
) -> list[str]:
    if recovery is None or clean is None:
        return []
    reasons = []
    if (
        clean.get("schema")
        != "bi100-recorded-session-cleanup-qualification-v1"
        or clean.get("version") != 1
        or clean.get("qualified") is not True
        or clean.get("reasons") != []
        or clean.get("emergency_recovery_used") is not False
        or clean.get("production_promotion_authorized") is not False
        or clean.get("input_sha256") != sha256(recovery_path)
    ):
        reasons.append("recorded-session cleanup did not qualify")
    if (
        recovery.get("schema") != "bi100-recorded-session-cleanup-v1"
        or recovery.get("version") != 1
        or recovery.get("qualified") is not True
        or recovery.get("reasons") != []
    ):
        reasons.append("recorded-session recovery contract differs")
    return reasons


def _postflight_reasons(value: Json | None) -> list[str]:
    if value is None:
        return []
    settling = value.get("settling") or {}
    if (
        value.get("schema") != "bi100-service-postflight-v1"
        or value.get("version") != 1
        or value.get("qualified") is not True
        or value.get("gpu_indices") != [0, 1, 2, 3]
        or value.get("missing_devices") != []
        or value.get("api_server_pids") != []
        or value.get("worker_pids") != []
        or value.get("gpu_processes") != []
        or value.get("scan_errors") != []
        or settling.get("timeout_s") != 30.0
        or settling.get("sample_interval_s") != 1.0
        or settling.get("required_clean_samples") != 3
        or not isinstance(settling.get("final_clean_streak"), int)
        or settling.get("final_clean_streak", 0) < 3
        or not isinstance(settling.get("attempts"), int)
        or settling.get("attempts", 0) < 3
    ):
        return ["orchestrator postflight did not prove a clean TP4 host"]
    privacy = value.get("privacy") or {}
    if (
        privacy.get("command_lines_recorded") is not False
        or privacy.get("environment_recorded") is not False
    ):
        return ["orchestrator postflight privacy contract differs"]
    return []


def _preflight_reasons(value: Json | None) -> list[str]:
    if value is None:
        return []
    results = value.get("results")
    if (
        value.get("schema") != "bi100-gpu-preflight-v1"
        or value.get("version") != 1
        or value.get("ok") is not True
        or value.get("gpus") != [0, 1, 2, 3]
        or value.get("matmul_size") != 1024
        or value.get("timeout_s") != 25.0
        or not isinstance(results, list)
        or len(results) != 4
    ):
        return ["orchestrator final GPU preflight did not qualify"]
    for expected_gpu, result in enumerate(results):
        if (
            not isinstance(result, dict)
            or result.get("gpu") != expected_gpu
            or result.get("ok") is not True
            or result.get("stage") != "done"
            or result.get("returncode") != 0
        ):
            return ["orchestrator final GPU preflight did not qualify"]
    return []


def _recompute_aggregate(root: Path) -> Json:
    return m1_137.compare_from_paths(
        control_root=root / "control",
        candidate_root=root / "candidate",
        score_comparison=root / "ifeval_score_comparison.json",
        exact_comparison=root / "ifeval_exact_comparison.json",
        paired_noninferiority=(
            root / "ifeval_paired_noninferiority.json"),
    )


def _record_set_digest(records: list[tuple[str, str]]) -> str:
    payload = json.dumps(
        records, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _rescan_current_artifacts(root: Path) -> tuple[list[str], Json]:
    reasons = []
    fatal_records: list[tuple[str, str]] = []
    fatal_match_count = 0
    try:
        log_paths = sorted(
            item
            for item in root.rglob("*")
            if item.is_file()
            and item.suffix in {".log", ".stdout", ".stderr"}
            and item.name not in {
                "final_qualification.stdout",
                "final_qualification.stderr",
            }
        )
    except OSError:
        log_paths = []
        reasons.append("final fatal rescan could not enumerate inputs")
    for path in log_paths:
        relative = path.relative_to(root).as_posix()
        try:
            digest_builder = hashlib.sha256()
            matched = False
            with path.open("rb") as stream:
                for line in stream:
                    digest_builder.update(line)
                    if FATAL_PATTERN.search(line):
                        matched = True
            digest = digest_builder.hexdigest()
        except OSError:
            reasons.append("final fatal rescan could not read an input")
            continue
        fatal_records.append((relative, digest))
        fatal_match_count += int(matched)
    if fatal_match_count:
        reasons.append(
            "final fatal rescan found "
            f"{fatal_match_count} affected files")

    rc_records: list[tuple[str, str]] = []
    timeout_count = 0
    malformed_count = 0
    try:
        rc_paths = sorted(root.rglob("*.rc"))
    except OSError:
        rc_paths = []
        reasons.append("final timeout rescan could not enumerate inputs")
    for path in rc_paths:
        if path.name == "final_qualification.rc":
            continue
        relative = path.relative_to(root).as_posix()
        try:
            payload = path.read_bytes()
            raw = payload.decode("utf-8").strip()
            digest = hashlib.sha256(payload).hexdigest()
        except (OSError, UnicodeError):
            malformed_count += 1
            continue
        rc_records.append((relative, digest))
        if not raw.isdigit():
            malformed_count += 1
        elif int(raw) in {124, 137, 143}:
            timeout_count += 1
    if malformed_count:
        reasons.append(
            "final timeout rescan found "
            f"{malformed_count} malformed return codes")
    if timeout_count:
        reasons.append(
            "final timeout rescan found "
            f"{timeout_count} timeout return codes")
    return reasons, {
        "fatal_input_file_count": len(fatal_records),
        "fatal_input_set_sha256": _record_set_digest(fatal_records),
        "fatal_match_file_count": fatal_match_count,
        "return_code_file_count": len(rc_records),
        "return_code_input_set_sha256": _record_set_digest(rc_records),
        "malformed_return_code_count": malformed_count,
        "timeout_return_code_count": timeout_count,
    }


def qualify(
    root: Path,
    aggregate_path: Path,
    *,
    aggregate_recomputer: Callable[[Path], Json] = _recompute_aggregate,
) -> Json:
    reasons: list[str] = []
    expected_aggregate = root / "aggregate.json"
    try:
        aggregate_is_bound = (
            aggregate_path.resolve(strict=True)
            == expected_aggregate.resolve(strict=True)
        )
    except OSError:
        aggregate_is_bound = False
    if not aggregate_is_bound:
        reasons.append("aggregate path is not bound to the run root")

    rcs = {
        name: _read_rc(root, name, reasons)
        for name in EXPECTED_RCS
    }
    aggregate = _load_json(aggregate_path, "aggregate", reasons)
    try:
        recomputed_aggregate = aggregate_recomputer(root)
    except Exception:
        recomputed_aggregate = None
        reasons.append(
            "M1-137 aggregate could not be recomputed from arm evidence")
    if (
        recomputed_aggregate is not None
        and aggregate != recomputed_aggregate
    ):
        reasons.append(
            "M1-137 aggregate differs from recomputed arm evidence")
    recovery_path = root / "orchestrator_recovery.json"
    recovery = _load_json(recovery_path, "recovery", reasons)
    recovery_clean = _load_json(
        root / "orchestrator_recovery_clean.json",
        "recovery qualification",
        reasons,
    )
    postflight = _load_json(
        root / "orchestrator_postflight.json", "postflight", reasons)
    preflight = _load_json(
        root / "orchestrator_preflight_after.json",
        "final preflight",
        reasons,
    )
    reasons.extend(_aggregate_reasons(aggregate))
    reasons.extend(_recovery_reasons(
        recovery, recovery_clean, recovery_path))
    reasons.extend(_postflight_reasons(postflight))
    reasons.extend(_preflight_reasons(preflight))
    rescan_reasons, rescan = _rescan_current_artifacts(root)
    reasons.extend(rescan_reasons)

    for name in ("orchestrator_fatal_scan.txt",
                 "orchestrator_timeout_scan.txt"):
        path = root / name
        try:
            payload = path.read_bytes()
        except OSError:
            reasons.append(f"{name} is missing")
        else:
            if payload:
                reasons.append(f"{name} is not empty")

    evidence = {}
    for name in EVIDENCE_FILES:
        path = root / name
        evidence[f"{name}_sha256"] = (
            sha256(path) if path.is_file() else None)
    for name in EXPECTED_RCS:
        path = root / f"{name}.rc"
        evidence[f"{name}.rc_sha256"] = (
            sha256(path) if path.is_file() else None)

    qualified = not reasons
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": qualified,
        "ifeval_two_point_capability_surface_authorized": qualified,
        "performance_authorized": False,
        "default_change_authorized": False,
        "yaml_change_authorized": False,
        "main_merge_authorized": False,
        "production_promotion_authorized": False,
        "lifecycle_return_codes": rcs,
        "evidence": evidence,
        "final_rescan": rescan,
        "reasons": reasons,
        "privacy": {
            "contains_raw_requests": False,
            "contains_raw_model_outputs": False,
            "contains_sample_outcomes": False,
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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = qualify(args.run_root.resolve(), args.aggregate)
    _atomic_write(args.out, report)
    print(json.dumps({
        "out": str(args.out),
        "qualified": report["qualified"],
        "reason_count": len(report["reasons"]),
    }, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
