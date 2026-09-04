#!/usr/bin/env python3
"""Run the adaptive attention-only M1-178 teacher-forced funnel."""

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

import compare_teacher_forced_logprobs_v2 as comparison  # noqa: E402


TARGETS = comparison.TARGETS
SCHEMA = "bi100-m1-178-attention-teacher-forced-v1"


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
    return {
        name: root / name for name in (
            "runner_status.json", "runtime_manifest.json", "measurement.json",
            "fatal_scan.json", "postflight_after.json", "scoped_cleanup.json",
        )
    }


def validate_arm(root: Path, selector: str) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    paths = _arm_paths(root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return ([f"{selector}: missing arm artifacts: {', '.join(missing)}"], {})
    try:
        values = {name: _load(path) for name, path in paths.items()}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ([f"{selector}: malformed arm artifact: {exc}"], {})
    status = values["runner_status.json"]
    manifest = values["runtime_manifest.json"]
    fatal = values["fatal_scan.json"]
    postflight = values["postflight_after.json"]
    expected_population = {
        "expected": 4, "attempted": 4, "completed": 4, "failed": 0,
    }
    if (status.get("schema") != "bi100-attention-operator-tp4-arm-v1"
            or status.get("version") != 1
            or status.get("workload_mode") != "teacher_forced"
            or status.get("selector") != selector
            or status.get("qualified") is not True
            or status.get("result_status") != "pass"
            or status.get("returncode") != 0
            or status.get("terminal_stage") != "complete"
            or status.get("targets") != list(TARGETS)
            or status.get("repetitions") != 1
            or status.get("request_population") != expected_population
            or status.get("service_startups") != 1
            or not isinstance(status.get("gates"), dict)
            or any(value != 0 for value in status.get("gates", {}).values())):
        reasons.append(f"{selector}: runner identity/population/lifecycle failed")
    dispatch = status.get("dispatch_count")
    if (selector == "candidate" and (
            not isinstance(dispatch, int) or dispatch <= 0)):
        reasons.append("candidate: fused-prefill dispatch is absent")
    if selector == "control" and dispatch != 0:
        reasons.append("control: candidate dispatch was observed")
    environment = manifest.get("environment")
    if (manifest.get("schema") != "bi100-attention-operator-runtime-v1"
            or manifest.get("version") != 1
            or manifest.get("workload_mode") != "teacher_forced"
            or manifest.get("tensor_parallel_size") != 4
            or manifest.get("dtype") != "float16"
            or manifest.get("max_model_len") != 262144
            or manifest.get("block_size") != 16
            or not isinstance(environment, dict)
            or environment.get("BI100_ATTN_COREX_FUSED_PREFILL")
            != ("1" if selector == "candidate" else "0")
            or environment.get("BI100_CACHE_TRACE") != "0"):
        reasons.append(f"{selector}: runtime/selector manifest failed")
    if (fatal.get("qualified") is not True
            or any((fatal.get("category_counts") or {}).values())):
        reasons.append(f"{selector}: fatal scan failed")
    if (postflight.get("qualified") is not True
            or postflight.get("api_server_pids")
            or postflight.get("worker_pids")
            or postflight.get("gpu_processes")):
        reasons.append(f"{selector}: postflight process cleanup failed")
    return reasons, values


def cross_arm_reasons(arms: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    labels = list(arms)
    reference_status = arms[labels[0]]["runner_status.json"]
    reference_manifest = arms[labels[0]]["runtime_manifest.json"]
    for label in labels[1:]:
        status = arms[label]["runner_status.json"]
        manifest = arms[label]["runtime_manifest.json"]
        for field in ("source_revision", "runtime_identity", "instance",
                      "model_path", "workload_id", "session_preflight_id",
                      "source_dirty_summary"):
            if not reference_status.get(field) or (
                    status.get(field) != reference_status.get(field)):
                reasons.append(f"{label}: cross-arm {field} differs or is empty")
        for field in ("source_revision", "runtime_identity", "instance",
                      "model_path", "tokenizer_path", "command"):
            if not reference_manifest.get(field) or (
                    manifest.get(field) != reference_manifest.get(field)):
                reasons.append(
                    f"{label}: cross-arm runtime {field} differs or is empty")
        left_environment = dict(reference_manifest.get("environment") or {})
        right_environment = dict(manifest.get("environment") or {})
        left_environment.pop("BI100_ATTN_COREX_FUSED_PREFILL", None)
        right_environment.pop("BI100_ATTN_COREX_FUSED_PREFILL", None)
        if left_environment != right_environment:
            reasons.append(f"{label}: environment differs beyond selector")
    return reasons


def _run_arm(
    args: argparse.Namespace,
    label: str,
    selector: str,
    environment: dict[str, str],
) -> int:
    command = [
        sys.executable, str(ROOT / "scripts/run_attention_operator_tp4_arm.py"),
        args.instance, str(args.run_root / label),
        "--selector", selector,
        "--pair-id", args.pair_id,
        "--session-preflight", str(args.session_preflight),
        "--targets", ",".join(map(str, TARGETS)),
        "--repetitions", "1",
        "--workload", "teacher_forced",
    ]
    print(json.dumps({"event": "arm_start", "arm": label,
                      "selector": selector}, sort_keys=True), flush=True)
    returncode = subprocess.run(
        command, env=environment, cwd=args.run_root / "runtime-workdir",
        check=False).returncode
    print(json.dumps({"event": "arm_end", "arm": label,
                      "returncode": returncode}, sort_keys=True), flush=True)
    return returncode


def _base_summary(
    args: argparse.Namespace,
    started: float,
    arms: dict[str, dict[str, Any]],
    quick: dict[str, Any],
    *,
    control_b_triggered: bool,
) -> dict[str, Any]:
    statuses = {
        label: value["runner_status.json"] for label, value in arms.items()
    }
    first = next(iter(statuses.values()), {})
    return {
        "schema": SCHEMA,
        "version": 1,
        "status": quick["status"],
        "classification": quick["classification"],
        "change_scope": "attention_operator",
        "source_revision": first.get("source_revision"),
        "source_dirty_summary": first.get("source_dirty_summary"),
        "runtime_identity": first.get("runtime_identity"),
        "instance": first.get("instance", args.instance),
        "model_path": first.get("model_path"),
        "pair_id": args.pair_id,
        "targets": list(TARGETS),
        "positions_per_request": 64,
        "quick_screen": quick,
        "control_b_triggered": control_b_triggered,
        "control_b_trigger_reason": (
            "control_a_candidate_quick_screen_passed"
            if control_b_triggered else "quick_screen_did_not_pass"),
        "service_startups": len(arms),
        "teacher_forced_requests": 4 * len(arms),
        "cached_tokens": {label: [
            case.get("cached_tokens")
            for case in value["measurement.json"].get("cases", [])
        ] for label, value in arms.items()},
        "dispatch": {
            label: status.get("dispatch_count")
            for label, status in statuses.items()
        },
        "wall_time_s": time.monotonic() - started,
        "old_l3_comparison": {
            "old_requests_for_three_arms": 228,
            "new_requests_if_control_b_runs": 12,
            "requests_eliminated_if_control_b_runs": 216,
            "request_reduction_fraction": 216 / 228,
            "old_service_startups": 3,
            "new_service_startups_if_control_b_runs": 3,
            "adaptive_early_stop_can_avoid_control_b": True,
        },
        "lifecycle": {
            label: {
                "fatal_scan_qualified": value["fatal_scan.json"].get(
                    "qualified"),
                "postflight_qualified": value["postflight_after.json"].get(
                    "qualified"),
                "cleanup_recorded": bool(value["scoped_cleanup.json"]),
            } for label, value in arms.items()
        },
        "privacy": {
            "contains_token_keys": False,
            "contains_token_ids": False,
            "contains_prompts": False,
            "contains_model_outputs": False,
            "contains_credentials": False,
        },
        "capability_run": False,
        "performance_rerun": False,
        "formal_881_run": False,
        "promotion_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--session-preflight", type=Path, required=True)
    parser.add_argument(
        "--contract", type=Path,
        default=ROOT / "quality/layered_quality_gate.v2.json")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    if args.run_root.exists() or not args.run_root.is_absolute():
        parser.error("run root must be a new absolute path")
    if not args.session_preflight.is_file() or not args.contract.is_file():
        parser.error("session preflight or v2 contract is missing")
    args.run_root.mkdir(parents=True)
    (args.run_root / "runtime-workdir").mkdir()
    started = time.monotonic()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{ROOT / 'tests'}:{ROOT / 'scripts'}:"
        f"{environment.get('PYTHONPATH', '')}")
    environment["BI100_TEACHER_FORCED_HMAC_KEY"] = secrets.token_hex(32)
    arms: dict[str, dict[str, Any]] = {}

    for label, selector in (("control_a", "control"),
                            ("candidate", "candidate")):
        rc = _run_arm(args, label, selector, environment)
        reasons, evidence = validate_arm(args.run_root / label, selector)
        if rc or reasons:
            status = "fail" if selector == "candidate" and evidence.get(
                "runner_status.json", {}).get("result_status") == "fail" else "invalid"
            quick = {
                "status": status,
                "classification": (
                    "candidate_hard_failure" if status == "fail"
                    else "invalid_evidence"),
                "reasons": reasons or [f"{label}: arm runner exited {rc}"],
                "control_b_authorized": False,
            }
            if evidence:
                arms[label] = evidence
            summary = _base_summary(
                args, started, arms, quick, control_b_triggered=False)
            _write(args.run_root / "summary.json", summary)
            environment.pop("BI100_TEACHER_FORCED_HMAC_KEY", None)
            return {"fail": 1, "invalid": 2}[status]
        arms[label] = evidence

    reasons = cross_arm_reasons(arms)
    if reasons:
        quick = {
            "status": "invalid", "classification": "invalid_evidence",
            "reasons": reasons, "control_b_authorized": False,
        }
    else:
        quick = comparison.quick_screen(
            arms["control_a"]["measurement.json"],
            arms["candidate"]["measurement.json"])
    _write(args.run_root / "quick_screen.json", quick)
    print(json.dumps({
        "event": "quick_screen",
        "status": quick["status"],
        "classification": quick["classification"],
        "control_b_authorized": quick.get("control_b_authorized", False),
    }, sort_keys=True), flush=True)
    if quick["status"] != "pass":
        summary = _base_summary(
            args, started, arms, quick, control_b_triggered=False)
        _write(args.run_root / "summary.json", summary)
        environment.pop("BI100_TEACHER_FORCED_HMAC_KEY", None)
        return {"inconclusive": 3, "fail": 1, "invalid": 2}[quick["status"]]

    rc = _run_arm(args, "control_b", "control", environment)
    reasons, evidence = validate_arm(args.run_root / "control_b", "control")
    if not rc and not reasons:
        arms["control_b"] = evidence
        reasons = cross_arm_reasons(arms)
    if rc or reasons:
        if evidence:
            arms["control_b"] = evidence
        invalid = {
            "status": "invalid", "classification": "invalid_evidence",
            "reasons": reasons or [f"control_b: arm runner exited {rc}"],
            "control_b_authorized": False,
        }
        summary = _base_summary(
            args, started, arms, invalid, control_b_triggered=True)
        _write(args.run_root / "summary.json", summary)
        environment.pop("BI100_TEACHER_FORCED_HMAC_KEY", None)
        return 2

    formal = comparison.compare(
        arms["control_a"]["measurement.json"],
        arms["control_b"]["measurement.json"],
        arms["candidate"]["measurement.json"],
        _load(args.contract),
    )
    _write(args.run_root / "distribution.json", formal)
    print(json.dumps({
        "event": "formal_distribution",
        "status": formal["status"],
        "classification": formal["classification"],
    }, sort_keys=True), flush=True)
    summary = _base_summary(
        args, started, arms, quick, control_b_triggered=True)
    summary["status"] = formal["status"]
    summary["classification"] = formal["classification"]
    summary["formal_distribution"] = formal
    summary["next_authorized_stage"] = (
        "small_capability_noninferiority" if formal["status"] == "pass"
        else "reviewer_adjudication")
    _write(args.run_root / "summary.json", summary)
    environment.pop("BI100_TEACHER_FORCED_HMAC_KEY", None)
    return {"pass": 0, "inconclusive": 3, "invalid": 2}[formal["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
