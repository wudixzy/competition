#!/usr/bin/env python3
"""Qualify a privacy-safe IFEval service run from its frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


STATUS_SCHEMA = "bi100-ifeval-service-gate-status-v1"
REPORT_SCHEMA = "bi100-ifeval-result-v1"
INSTALL_SCHEMA = "bi100-ifeval-offline-environment-v1"
QUALIFICATION_SCHEMA = "bi100-ifeval-service-qualification-v1"
EXPECTED_MANIFEST_SHA256 = (
    "07ec4efb5fe7afaacb55723c1d53be4c2f58c840bbd6a54bf944e15cfbca1855"
)
EXPECTED_GATES = (
    "runtime_identity", "runtime_contract", "prefix_allocator",
    "gdn_action_broadcast", "preflight_before", "startup",
    "startup_contract", "ifeval", "cleanup", "fatal_scan",
    "checkpoint_cleanup", "preflight_after", "preflight_comparison",
)
EXPECTED_ARTIFACTS = (
    "runtime_identity.json", "runtime_contract.json",
    "startup_contract.json", "ifeval_report.json", "fatal_scan.txt",
    "ifeval_progress.json",
    "preflight_before.json", "preflight_after.json",
    "preflight_comparison.json",
)
Json = dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
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


def load_json(path: Path, reasons: list[str]) -> Json:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reasons.append(f"cannot load {path.name}: {type(exc).__name__}")
        return {}
    if not isinstance(value, dict):
        reasons.append(f"{path.name} root must be an object")
        return {}
    return value


def read_rc(path: Path, reasons: list[str]) -> int | None:
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError) as exc:
        reasons.append(f"cannot read {path.name}: {type(exc).__name__}")
        return None
    return value


def _valid_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def qualify(
    run_root: Path,
    install_path: Path,
    expected_runtime_revision: str,
    expected_evaluator_revision: str,
) -> Json:
    reasons: list[str] = []
    status_path = run_root / "status.json"
    report_path = run_root / "ifeval_report.json"
    status = load_json(status_path, reasons)
    report = load_json(report_path, reasons)
    progress = load_json(run_root / "ifeval_progress.json", reasons)
    install = load_json(install_path, reasons)

    if (status.get("schema") != STATUS_SCHEMA
            or status.get("version") != 1
            or status.get("overall_rc") != 0):
        reasons.append("service status schema, version, or overall RC differs")
    if status.get("runtime_source_revision") != expected_runtime_revision:
        reasons.append("runtime source revision differs")
    if status.get("evaluator_source_revision") != expected_evaluator_revision:
        reasons.append("evaluator source revision differs")

    status_gates = status.get("gates") or {}
    if set(status_gates) != set(EXPECTED_GATES):
        reasons.append("service gate identities differ")
    gate_rc = {}
    for name in EXPECTED_GATES:
        value = read_rc(run_root / f"{name}.rc", reasons)
        gate_rc[name] = value
        if value != 0 or status_gates.get(name) != value:
            reasons.append(f"service gate failed or status drifted: {name}")
    overall_rc = read_rc(run_root / "overall.rc", reasons)
    if overall_rc != 0:
        reasons.append("overall.rc is not zero")

    status_artifacts = status.get("artifacts") or {}
    if set(status_artifacts) != set(EXPECTED_ARTIFACTS):
        reasons.append("service artifact identities differ")
    artifact_sha256 = {}
    for name in EXPECTED_ARTIFACTS:
        path = run_root / name
        if not path.is_file():
            reasons.append(f"service artifact is missing: {name}")
            continue
        digest = sha256(path)
        artifact_sha256[name] = digest
        if not _valid_digest(status_artifacts.get(name)):
            reasons.append(f"status artifact digest is invalid: {name}")
        elif status_artifacts[name] != digest:
            reasons.append(f"status artifact digest differs: {name}")

    if ((run_root / "fatal_scan.txt").is_file()
            and (run_root / "fatal_scan.txt").stat().st_size != 0):
        reasons.append("fatal scan is not empty")
    if (run_root / "ifeval.checkpoint.json").exists():
        reasons.append("raw IFEval checkpoint was retained")
    status_privacy = status.get("privacy") or {}
    if (status_privacy.get("raw_service_log_outside_repository") is not True
            or status_privacy.get("raw_checkpoint_absent_after_lifecycle")
            is not True
            or status_privacy.get("contains_credentials") is not False):
        reasons.append("service status privacy contract differs")

    if (report.get("schema") != REPORT_SCHEMA
            or report.get("version") != 1
            or report.get("qualified") is not True
            or report.get("quality_run_eligible_for_baseline") is not True
            or report.get("promotion_authorized") is not False):
        reasons.append("IFEval report is not baseline eligible")
    manifest = report.get("manifest") or {}
    if (manifest.get("sha256") != EXPECTED_MANIFEST_SHA256
            or manifest.get("full_selection") is not True
            or len(manifest.get("selected_keys") or []) != 64):
        reasons.append("IFEval manifest identity or selection differs")
    transport = report.get("transport") or {}
    if transport != {"selected": 64, "completed": 64, "errors": 0}:
        reasons.append("IFEval transport is incomplete")
    cases = report.get("cases") or []
    if (len(cases) != 64
            or any(case.get("status") != "pass" for case in cases)):
        reasons.append("IFEval cases are incomplete")
    summary = report.get("summary") or {}
    if (summary.get("prompt_total") != 64
            or not isinstance(summary.get("instruction_total"), int)
            or summary.get("instruction_total", 0) <= 0):
        reasons.append("IFEval score summary is incomplete")
    report_runtime = report.get("runtime") or {}
    expected_report_optimization = dict(status.get("optimization") or {})
    if expected_report_optimization.get("fused_prefill") in ("0", "1"):
        expected_report_optimization["fused_prefill"] = (
            expected_report_optimization["fused_prefill"] == "1")
    if (report_runtime.get("source_revision") != expected_runtime_revision
            or report_runtime.get("gpu_count") != 4
            or report_runtime.get("tensor_parallel_size") != 4
            or report_runtime.get("max_model_len") != 262144
            or report_runtime.get("optimization")
            != expected_report_optimization):
        reasons.append("IFEval runtime contract differs from service status")
    report_privacy = report.get("privacy") or {}
    if (any(report_privacy.get(name) is not False for name in (
            "contains_credentials", "contains_raw_prompts",
            "contains_raw_model_outputs", "contains_reasoning_text"))
            or report_privacy.get("checkpoint_deleted") is not True):
        reasons.append("IFEval report privacy contract differs")
    if (progress.get("schema") != "bi100-ifeval-progress-v1"
            or progress.get("version") != 1
            or progress.get("run_id_sha256") != report.get("run_id_sha256")
            or progress.get("selected") != 64
            or progress.get("attempted") != 64
            or progress.get("successful") != 64
            or progress.get("errors") != 0
            or progress.get("last_ordinal") != 64
            or progress.get("complete") is not True
            or progress.get("report_sha256") != sha256(report_path)
            or progress.get("failures") != []):
        reasons.append("IFEval progress is incomplete or differs")
    progress_privacy = progress.get("privacy") or {}
    if any(progress_privacy.get(name) is not False for name in (
            "contains_credentials", "contains_raw_prompts",
            "contains_raw_model_outputs", "contains_reasoning_text")):
        reasons.append("IFEval progress privacy contract differs")
    runtime_contract = report.get("runtime_contract") or {}
    if (runtime_contract.get("file_sha256")
            != artifact_sha256.get("runtime_contract.json")):
        reasons.append("IFEval runtime-contract file digest differs")

    if (install.get("schema") != INSTALL_SCHEMA
            or install.get("version") != 1
            or install.get("qualified") is not True
            or install.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
            or not str(install.get("python", "")).startswith("3.10.")
            or install.get("system_site_packages_modified") is not False):
        reasons.append("offline IFEval environment identity differs")
    preflight = load_json(run_root / "preflight_comparison.json", reasons)
    if preflight.get("qualified") is not True:
        reasons.append("GPU preflight comparison is not qualified")

    optimization = status.get("optimization") or {}
    return {
        "schema": QUALIFICATION_SCHEMA,
        "version": 1,
        "qualified": not reasons,
        "promotion_authorized": False,
        "reasons": reasons,
        "source": {
            "runtime_revision": status.get("runtime_source_revision"),
            "evaluator_revision": status.get("evaluator_source_revision"),
        },
        "optimization": optimization,
        "quality_summary": summary,
        "transport": transport,
        "lifecycle": {
            "overall_rc": overall_rc,
            "gates": gate_rc,
            "fatal_scan_empty": (
                (run_root / "fatal_scan.txt").is_file()
                and (run_root / "fatal_scan.txt").stat().st_size == 0),
        },
        "artifact_sha256": artifact_sha256,
        "input_sha256": {
            "status.json": sha256(status_path) if status_path.is_file() else None,
            "ifeval_report.json": (
                sha256(report_path) if report_path.is_file() else None),
            "ifeval_install.json": (
                sha256(install_path) if install_path.is_file() else None),
        },
        "privacy": {
            "contains_credentials": False,
            "contains_raw_prompts": False,
            "contains_raw_model_outputs": False,
            "contains_reasoning_text": False,
            "raw_service_log_included": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ifeval-install", type=Path, required=True)
    parser.add_argument("--expected-runtime-revision", required=True)
    parser.add_argument("--expected-evaluator-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"qualification output already exists: {args.out}")
    report = qualify(
        args.run_root,
        args.ifeval_install,
        args.expected_runtime_revision,
        args.expected_evaluator_revision,
    )
    atomic_write(args.out, report)
    print(json.dumps({
        "out": str(args.out),
        "qualified": report["qualified"],
        "reasons": report["reasons"],
        "sha256": sha256(args.out),
    }, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
