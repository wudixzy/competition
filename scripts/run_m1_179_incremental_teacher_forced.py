#!/usr/bin/env python3
"""Run the fixed M1-109/M1-162/M1-109 TP4 distribution attribution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import compare_m1_179_teacher_forced as comparison  # noqa: E402


SCHEMA = "bi100-m1-179-incremental-teacher-forced-run-v1"
TARGETS = comparison.TARGETS
ARM_VARIANTS = comparison.EXPECTED_VARIANTS
ARM_SELECTORS = {"control_a": "control", "candidate": "candidate",
                 "control_b": "control"}


def control_b_required(first_two_arms_valid: bool,
                       distribution_drift_observed: bool) -> bool:
    """Drift is a reason to calibrate A/A, not a reason to skip it."""
    del distribution_drift_observed
    return first_two_arms_valid


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def _arm_paths(root: Path) -> dict[str, Path]:
    return {name: root / name for name in (
        "runner_status.json", "runtime_manifest.json", "measurement.json",
        "fatal_scan.json", "postflight_after.json", "scoped_cleanup.json",
    )}


def validate_arm(root: Path, label: str,
                 extension_path: Path,
                 extension_sha256: str) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    paths = _arm_paths(root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return ([f"{label}: missing arm artifacts: {', '.join(missing)}"], {})
    try:
        values = {name: _load(path) for name, path in paths.items()}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ([f"{label}: malformed arm artifact: {exc}"], {})
    status = values["runner_status.json"]
    manifest = values["runtime_manifest.json"]
    fatal = values["fatal_scan.json"]
    postflight = values["postflight_after.json"]
    variant = ARM_VARIANTS[label]
    selector = ARM_SELECTORS[label]
    expected_population = {
        "expected": 4, "attempted": 4, "completed": 4, "failed": 0,
    }
    if (status.get("schema") != "bi100-attention-operator-tp4-arm-v1"
            or status.get("version") != 1
            or status.get("workload_mode") != "teacher_forced"
            or status.get("selector") != selector
            or status.get("fused_variant") != variant
            or status.get("extension_path") != str(extension_path)
            or status.get("extension_sha256") != extension_sha256
            or status.get("qualified") is not True
            or status.get("result_status") != "pass"
            or status.get("returncode") != 0
            or status.get("terminal_stage") != "complete"
            or status.get("targets") != list(TARGETS)
            or status.get("repetitions") != 1
            or status.get("request_population") != expected_population
            or status.get("service_startups") != 1
            or not isinstance(status.get("dispatch_count"), int)
            or status["dispatch_count"] <= 0
            or not isinstance(status.get("gates"), dict)
            or any(value != 0 for value in status.get("gates", {}).values())):
        reasons.append(f"{label}: runner/variant/population/lifecycle failed")
    environment = manifest.get("environment")
    extension = manifest.get("extension_identity")
    if (manifest.get("schema") != "bi100-attention-operator-runtime-v1"
            or manifest.get("version") != 1
            or manifest.get("workload_mode") != "teacher_forced"
            or manifest.get("fused_variant") != variant
            or manifest.get("tensor_parallel_size") != 4
            or manifest.get("dtype") != "float16"
            or manifest.get("max_model_len") != 262144
            or manifest.get("block_size") != 16
            or not isinstance(environment, dict)
            or environment.get("BI100_ATTN_COREX_FUSED_PREFILL") != "1"
            or environment.get(
                "BI100_ATTN_COREX_FUSED_PREFILL_VARIANT") != variant
            or environment.get(
                "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION")
            != str(extension_path)
            or environment.get(
                "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256")
            != extension_sha256
            or environment.get("BI100_CACHE_TRACE") != "0"
            or extension != {
                "module_path": str(extension_path),
                "runtime_loaded_module": str(extension_path),
                "sha256": extension_sha256,
            }):
        reasons.append(f"{label}: runtime extension identity failed")
    if fatal.get("qualified") is not True or any(
            (fatal.get("category_counts") or {}).values()):
        reasons.append(f"{label}: fatal scan failed")
    if (postflight.get("qualified") is not True
            or postflight.get("api_server_pids")
            or postflight.get("worker_pids")
            or postflight.get("gpu_processes")):
        reasons.append(f"{label}: postflight cleanup failed")
    return reasons, values


def cross_arm_reasons(arms: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    reference_status = arms["control_a"]["runner_status.json"]
    reference_manifest = arms["control_a"]["runtime_manifest.json"]
    ignored_environment = {
        "BI100_ATTN_COREX_FUSED_PREFILL_VARIANT",
        "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION",
        "BI100_ATTN_COREX_FUSED_PREFILL_EXTENSION_SHA256",
    }
    for label in ("candidate", "control_b"):
        if label not in arms:
            continue
        status = arms[label]["runner_status.json"]
        manifest = arms[label]["runtime_manifest.json"]
        for field in ("source_revision", "runtime_identity", "instance",
                      "model_path", "workload_id", "session_preflight_id",
                      "source_dirty_summary"):
            if not reference_status.get(field) or status.get(field) != reference_status.get(field):
                reasons.append(f"{label}: cross-arm {field} differs")
        for field in ("source_revision", "runtime_identity", "instance",
                      "model_path", "tokenizer_path", "command", "compiler"):
            if not reference_manifest.get(field) or manifest.get(field) != reference_manifest.get(field):
                reasons.append(f"{label}: cross-arm runtime {field} differs")
        left = dict(reference_manifest.get("environment") or {})
        right = dict(manifest.get("environment") or {})
        for name in ignored_environment:
            left.pop(name, None)
            right.pop(name, None)
        if left != right:
            reasons.append(f"{label}: environment differs beyond extension variant")
    return reasons


def _run_arm(args: argparse.Namespace, label: str,
             environment: dict[str, str]) -> int:
    variant = ARM_VARIANTS[label]
    extension_path = (args.m1_162_extension if label == "candidate"
                      else args.m1_109_extension)
    extension_sha256 = (args.m1_162_sha256 if label == "candidate"
                        else args.m1_109_sha256)
    command = [
        sys.executable, str(ROOT / "scripts/run_attention_operator_tp4_arm.py"),
        args.instance, str(args.run_root / label),
        "--selector", ARM_SELECTORS[label],
        "--pair-id", args.pair_id,
        "--session-preflight", str(args.session_preflight),
        "--targets", ",".join(map(str, TARGETS)),
        "--repetitions", "1", "--workload", "teacher_forced",
        "--fused-variant", variant,
        "--extension-path", str(extension_path),
        "--extension-sha256", extension_sha256,
    ]
    print(json.dumps({"event": "arm_start", "arm": label,
                      "variant": variant}, sort_keys=True), flush=True)
    returncode = subprocess.run(
        command, env=environment, cwd=args.run_root / "runtime-workdir",
        check=False).returncode
    print(json.dumps({"event": "arm_end", "arm": label,
                      "variant": variant, "returncode": returncode},
                     sort_keys=True), flush=True)
    return returncode


def _summary(args: argparse.Namespace, started: float,
             arms: dict[str, dict[str, Any]], status: str,
             classification: str, reasons: list[str],
             distribution: dict[str, Any] | None = None) -> dict[str, Any]:
    statuses = {label: values["runner_status.json"]
                for label, values in arms.items()}
    first = next(iter(statuses.values()), {})
    return {
        "schema": SCHEMA, "version": 1,
        "status": status, "classification": classification,
        "reasons": reasons,
        "source_revision": first.get("source_revision"),
        "source_dirty_summary": first.get("source_dirty_summary"),
        "runtime_identity": first.get("runtime_identity"),
        "instance": first.get("instance", args.instance),
        "model_path": first.get("model_path"),
        "pair_id": args.pair_id,
        "arm_variants": ARM_VARIANTS,
        "all_arms_fused_prefill_enabled": True,
        "targets": list(TARGETS), "positions_per_request": 64,
        "service_startups": len(arms),
        "teacher_forced_model_requests": 4 * len(arms),
        "control_b_rule": (
            "run_after_any_valid_control_a_and_candidate_pair_regardless_of_drift"),
        "distribution": distribution,
        "cached_tokens": {label: [
            case.get("cached_tokens")
            for case in values["measurement.json"].get("cases", [])]
            for label, values in arms.items()},
        "dispatch": {label: {
            "variant": status_value.get("fused_variant"),
            "count": status_value.get("dispatch_count")}
            for label, status_value in statuses.items()},
        "wall_time_s": time.monotonic() - started,
        "lifecycle": {label: {
            "fatal_scan_qualified": values["fatal_scan.json"].get("qualified"),
            "postflight_qualified": values["postflight_after.json"].get("qualified"),
            "cleanup_recorded": bool(values["scoped_cleanup.json"]),
        } for label, values in arms.items()},
        "conclusions": {
            "experiment_validity": status != "invalid",
            "operator_numerics": "M1-176 pass retained; not rerun",
            "capability": "not run",
            "performance": "historical evidence retained; not rerun",
            "promotion": "not authorized",
        },
        "privacy": {
            "contains_token_keys": False, "contains_token_ids": False,
            "contains_prompts": False, "contains_model_outputs": False,
            "contains_credentials": False,
        },
        "promotion_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--session-preflight", type=Path, required=True)
    parser.add_argument("--m1-109-extension", type=Path, required=True)
    parser.add_argument("--m1-109-sha256", required=True)
    parser.add_argument("--m1-162-extension", type=Path, required=True)
    parser.add_argument("--m1-162-sha256", required=True)
    parser.add_argument("--contract", type=Path,
                        default=ROOT / "quality/layered_quality_gate.v2.json")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.m1_109_extension = args.m1_109_extension.resolve()
    args.m1_162_extension = args.m1_162_extension.resolve()
    if args.run_root.exists() or not args.run_root.is_absolute():
        parser.error("run root must be a new absolute path")
    if not args.session_preflight.is_file() or not args.contract.is_file():
        parser.error("preflight or v2 contract is missing")
    args.run_root.mkdir(parents=True)
    (args.run_root / "runtime-workdir").mkdir()
    started = time.monotonic()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{ROOT / 'tests'}:{ROOT / 'scripts'}:"
        f"{environment.get('PYTHONPATH', '')}")
    environment["BI100_TEACHER_FORCED_HMAC_KEY"] = secrets.token_hex(32)
    arms: dict[str, dict[str, Any]] = {}
    try:
        for label in ("control_a", "candidate", "control_b"):
            rc = _run_arm(args, label, environment)
            extension_path = (args.m1_162_extension if label == "candidate"
                              else args.m1_109_extension)
            extension_sha256 = (args.m1_162_sha256 if label == "candidate"
                                else args.m1_109_sha256)
            reasons, evidence = validate_arm(
                args.run_root / label, label,
                extension_path, extension_sha256)
            if evidence:
                arms[label] = evidence
            if rc or reasons:
                summary = _summary(
                    args, started, arms, "invalid", "invalid_evidence",
                    reasons or [f"{label}: arm runner exited {rc}"])
                _write(args.run_root / "summary.json", summary)
                return 2
            if label == "candidate":
                binding_reasons = cross_arm_reasons(arms)
                if binding_reasons:
                    summary = _summary(
                        args, started, arms, "invalid", "invalid_evidence",
                        binding_reasons)
                    _write(args.run_root / "summary.json", summary)
                    return 2
                if not control_b_required(True, True):
                    raise AssertionError("valid drift must require control B")
                print(json.dumps({
                    "event": "control_b_required",
                    "reason": "first_two_arms_valid_drift_does_not_stop_aa",
                }, sort_keys=True), flush=True)

        reasons = cross_arm_reasons(arms)
        if reasons:
            summary = _summary(args, started, arms, "invalid",
                               "invalid_evidence", reasons)
            _write(args.run_root / "summary.json", summary)
            return 2
        distribution = comparison.compare(
            arms["control_a"]["measurement.json"],
            arms["control_b"]["measurement.json"],
            arms["candidate"]["measurement.json"],
            _load(args.contract),
        )
        _write(args.run_root / "distribution.json", distribution)
        summary = _summary(
            args, started, arms, distribution["status"],
            distribution["classification"],
            distribution.get("reasons", []), distribution)
        _write(args.run_root / "summary.json", summary)
        print(json.dumps({
            "event": "distribution", "status": distribution["status"],
            "classification": distribution["classification"],
        }, sort_keys=True), flush=True)
        return {"pass": 0, "inconclusive": 3, "invalid": 2}[
            distribution["status"]]
    finally:
        environment.pop("BI100_TEACHER_FORCED_HMAC_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
