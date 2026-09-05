#!/usr/bin/env python3
"""Orchestrate resumable fused-off/M1-109 IFEval and distribution evidence."""

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
import compare_m1_181_ifeval as comparison  # noqa: E402


ARM_CONFIG = {
    "fused_off": ("control", None),
    "m1_109": ("control", "m1_109_fp32_qk"),
    "fused_off_b": ("control", None),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def _write(path: Path, value: dict[str, Any], mode: int = 0o644) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def arm_command(args: argparse.Namespace, label: str) -> list[str]:
    selector, variant = ARM_CONFIG[label]
    command = [
        sys.executable, str(ROOT / "scripts/run_attention_operator_tp4_arm.py"),
        args.instance, str(args.run_root / label), "--selector", selector,
        "--pair-id", args.pair_id, "--session-preflight",
        str(args.session_preflight), "--targets", "4096,16384,32768,65536",
        "--repetitions", "1", "--workload", "m1_181",
        "--arm-label", label,
    ]
    if variant is not None:
        command.extend(["--fused-variant", variant, "--extension-path",
                        str(args.m1_109_extension), "--extension-sha256",
                        args.m1_109_sha256])
    if label == "m1_109":
        command.extend(["--reference-fused-off",
                        str(args.run_root / "fused_off/measurement.json")])
    return command


def validate_arm(args: argparse.Namespace, label: str) -> tuple[list[str], dict[str, Any]]:
    root = args.run_root / label
    required = ("runner_status.json", "runtime_manifest.json",
                "measurement.json", "fatal_scan.json",
                "postflight_after.json", "scoped_cleanup.json")
    if any(not (root / name).is_file() for name in required):
        return [f"{label}: required arm artifact missing"], {}
    values = {name: _load(root / name) for name in required}
    status, manifest = values["runner_status.json"], values["runtime_manifest.json"]
    measurement = values["measurement.json"]
    selector, variant = ARM_CONFIG[label]
    reasons = comparison.arm_reasons(measurement, label)
    dispatch = status.get("dispatch_count")
    if (status.get("qualified") is not True
            or status.get("result_status") != "pass"
            or status.get("workload_mode") != "m1_181"
            or status.get("selector") != selector
            or status.get("fused_variant") != variant
            or status.get("algorithm_variant") != label
            or status.get("service_startups") != 1
            or status.get("returncode") != 0
            or not isinstance(dispatch, int)
            or (label == "m1_109" and dispatch <= 0)
            or (label != "m1_109" and dispatch != 0)):
        reasons.append(f"{label}: runner status or dispatch differs")
    extension = manifest.get("extension_identity")
    if label == "m1_109":
        expected = {"module_path": str(args.m1_109_extension),
                    "runtime_loaded_module": str(args.m1_109_extension),
                    "sha256": args.m1_109_sha256}
        if extension != expected:
            reasons.append("m1_109: loaded extension identity differs")
    elif extension is not None:
        reasons.append(f"{label}: extension must be absent")
    fatal = values["fatal_scan.json"]
    postflight = values["postflight_after.json"]
    if fatal.get("qualified") is not True or any(
            (fatal.get("category_counts") or {}).values()):
        reasons.append(f"{label}: fatal scan differs")
    if (postflight.get("qualified") is not True
            or postflight.get("api_server_pids")
            or postflight.get("worker_pids")
            or postflight.get("gpu_processes")):
        reasons.append(f"{label}: postflight differs")
    return reasons, values


def validate_inputs(args: argparse.Namespace) -> None:
    for path in (args.session_preflight, args.m1_109_extension,
                 args.numeric_summary):
        if not path.is_file():
            raise ValueError(f"required input missing: {path}")
    numeric = _load(args.numeric_summary)
    if (numeric.get("schema") != "bi100-m1-181-m1-109-numeric-v1"
            or numeric.get("status") != "pass"):
        raise ValueError("M1-109 numeric prerequisite did not pass")
    if (len(args.m1_109_sha256) != 64
            or any(char not in "0123456789abcdef"
                   for char in args.m1_109_sha256)):
        raise ValueError("M1-109 extension identity is invalid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--session-preflight", type=Path, required=True)
    parser.add_argument("--m1-109-extension", type=Path, required=True)
    parser.add_argument("--m1-109-sha256", required=True)
    parser.add_argument("--numeric-summary", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.m1_109_extension = args.m1_109_extension.resolve()
    validate_inputs(args)
    plan = {"schema": "bi100-m1-181-run-plan-v1", "version": 1,
            "arm_order": list(ARM_CONFIG),
            "commands": {label: arm_command(args, label)
                         for label in ARM_CONFIG},
            "control_b_condition": "numeric_pass_and_ifeval64_pass"}
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    args.run_root.mkdir(parents=True, exist_ok=True)
    runtime_workdir = args.run_root / "runtime-workdir"
    runtime_workdir.mkdir(exist_ok=True)
    _write(args.run_root / "plan.json", plan)
    key_path = args.run_root / "teacher_identity.key"
    if not key_path.exists():
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(secrets.token_hex(32))
            stream.flush()
            os.fsync(stream.fileno())
    if key_path.stat().st_mode & 0o077:
        raise ValueError("teacher identity key permissions differ")
    key = key_path.read_text(encoding="ascii")
    if len(key) != 64:
        raise ValueError("teacher identity key is invalid")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{ROOT / 'tests'}:{ROOT / 'scripts'}:"
        f"{environment.get('PYTHONPATH', '')}")
    environment["BI100_TEACHER_FORCED_HMAC_KEY"] = key
    started = time.monotonic()
    launched = 0
    arms: dict[str, dict[str, Any]] = {}
    reasons = []
    _write(args.run_root / "orchestrator_status.json", {
        "schema": "bi100-m1-181-orchestrator-status-v1", "version": 1,
        "state": "running", "completed_arms": [], "service_startups": 0})
    for label in ("fused_off", "m1_109"):
        existing_reasons, evidence = validate_arm(args, label)
        if not existing_reasons:
            arms[label] = evidence
            continue
        if (args.run_root / label).exists():
            reasons.extend(existing_reasons)
            break
        launched += 1
        rc = subprocess.run(arm_command(args, label), env=environment,
                            cwd=runtime_workdir, check=False).returncode
        arm_reasons, evidence = validate_arm(args, label)
        if rc or arm_reasons:
            reasons.extend(arm_reasons or [f"{label}: runner rc={rc}"])
            break
        arms[label] = evidence
        _write(args.run_root / "orchestrator_status.json", {
            "schema": "bi100-m1-181-orchestrator-status-v1", "version": 1,
            "state": "running", "completed_arms": list(arms),
            "service_startups": launched})
    preliminary = None
    if not reasons and len(arms) == 2:
        preliminary = comparison.compare(
            arms["fused_off"]["measurement.json"],
            arms["m1_109"]["measurement.json"])
        run_b = (preliminary.get("ifeval_statistical_capability", {}).get(
            "status") == "pass")
        if run_b:
            label = "fused_off_b"
            existing_reasons, evidence = validate_arm(args, label)
            if not existing_reasons:
                arms[label] = evidence
            elif (args.run_root / label).exists():
                reasons.extend(existing_reasons)
            else:
                launched += 1
                rc = subprocess.run(arm_command(args, label), env=environment,
                                    cwd=runtime_workdir, check=False).returncode
                arm_reasons, evidence = validate_arm(args, label)
                if rc or arm_reasons:
                    reasons.extend(arm_reasons or [f"{label}: runner rc={rc}"])
                else:
                    arms[label] = evidence
    postflight_path = args.run_root / "final_postflight.json"
    postflight_rc = subprocess.run([
        sys.executable, str(ROOT / "tests/service_postflight_gate.py"),
        "--gpus", "0,1,2,3", "--out", str(postflight_path),
    ], env=environment, cwd=runtime_workdir, check=False).returncode
    postflight = _load(postflight_path) if postflight_path.is_file() else {}
    if postflight_rc or postflight.get("qualified") is not True:
        reasons.append("final service/GPU postflight failed")
    if reasons or len(arms) < 2:
        result = {"schema": "bi100-m1-181-run-v1", "version": 1,
                  "status": "invalid", "classification": "invalid_evidence",
                  "reasons": reasons, "promotion_authorized": False}
        rc = 2
    else:
        result = comparison.compare(
            arms["fused_off"]["measurement.json"],
            arms["m1_109"]["measurement.json"],
            arms.get("fused_off_b", {}).get("measurement.json"))
        rc = {"inconclusive": 3, "fail": 1, "invalid": 2}[result["status"]]
    result["experiment"] = {
        "service_startups_this_invocation": launched,
        "valid_arm_count": len(arms), "arm_order": list(arms),
        "wall_s": time.monotonic() - started,
        "invalid_retry_count": 0, "final_postflight": postflight,
    }
    _write(args.run_root / "summary.json", result)
    key_path.unlink(missing_ok=True)
    _write(args.run_root / "orchestrator_status.json", {
        "schema": "bi100-m1-181-orchestrator-status-v1", "version": 1,
        "state": "complete", "completed_arms": list(arms),
        "service_startups": launched, "summary_status": result["status"],
        "identity_key_deleted": not key_path.exists()})
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
