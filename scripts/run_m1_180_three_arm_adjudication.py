#!/usr/bin/env python3
"""Orchestrate fused-off/M1-109/M1-162 capability and distribution evidence."""

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

import compare_m1_180_adjudication as comparison  # noqa: E402


SCHEMA = "bi100-m1-180-three-arm-run-v1"
ARM_CONFIG = {
    "fused_off": ("control", None),
    "m1_109": ("control", "m1_109_fp32_qk"),
    "m1_162": ("candidate", "m1_162_fp16_qk"),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def arm_command(args: argparse.Namespace, label: str) -> list[str]:
    selector, variant = ARM_CONFIG[label]
    command = [
        sys.executable, str(ROOT / "scripts/run_attention_operator_tp4_arm.py"),
        args.instance, str(args.run_root / label),
        "--selector", selector, "--pair-id", args.pair_id,
        "--session-preflight", str(args.session_preflight),
        "--targets", "4096,16384,32768,65536", "--repetitions", "1",
        "--workload", "m1_180", "--arm-label", label,
    ]
    if variant is not None:
        extension = (args.m1_109_extension if label == "m1_109"
                     else args.m1_162_extension)
        digest = (args.m1_109_sha256 if label == "m1_109"
                  else args.m1_162_sha256)
        command.extend(["--fused-variant", variant,
                        "--extension-path", str(extension),
                        "--extension-sha256", digest])
    if label == "m1_162":
        command.extend([
            "--reference-fused-off",
            str(args.run_root / "fused_off/measurement.json"),
            "--reference-m1-109",
            str(args.run_root / "m1_109/measurement.json"),
        ])
    return command


def validate_arm(args: argparse.Namespace, label: str) -> tuple[list[str], dict[str, Any]]:
    root = args.run_root / label
    names = ("runner_status.json", "runtime_manifest.json", "measurement.json",
             "fatal_scan.json", "postflight_after.json", "scoped_cleanup.json")
    if any(not (root / name).is_file() for name in names):
        return [f"{label}: required arm artifact missing"], {}
    values = {name: _load(root / name) for name in names}
    status = values["runner_status.json"]
    manifest = values["runtime_manifest.json"]
    measurement = values["measurement.json"]
    fatal = values["fatal_scan.json"]
    postflight = values["postflight_after.json"]
    selector, variant = ARM_CONFIG[label]
    reasons = comparison.arm_reasons(measurement, label)
    if (status.get("qualified") is not True
            or status.get("result_status") != "pass"
            or status.get("workload_mode") != "m1_180"
            or status.get("algorithm_variant") != label
            or status.get("selector") != selector
            or status.get("fused_variant") != variant
            or status.get("service_startups") != 1
            or status.get("returncode") != 0
            or status.get("request_population", {}).get("failed") != 0
            or not isinstance(status.get("dispatch_count"), int)
            or (label == "fused_off" and status["dispatch_count"] != 0)
            or (label != "fused_off" and status["dispatch_count"] <= 0)):
        reasons.append(f"{label}: runner status/dispatch differs")
    environment = manifest.get("environment") or {}
    if (manifest.get("algorithm_variant") != label
            or manifest.get("tensor_parallel_size") != 4
            or manifest.get("dtype") != "float16"
            or manifest.get("max_model_len") != 262144
            or manifest.get("block_size") != 16
            or environment.get("BI100_ATTN_COREX_FUSED_PREFILL")
            != ("0" if label == "fused_off" else "1")):
        reasons.append(f"{label}: runtime selector/shape differs")
    extension = manifest.get("extension_identity")
    if label == "fused_off":
        if extension is not None:
            reasons.append("fused_off: extension must be absent")
    else:
        path = args.m1_109_extension if label == "m1_109" else args.m1_162_extension
        digest = args.m1_109_sha256 if label == "m1_109" else args.m1_162_sha256
        if extension != {"module_path": str(path),
                         "runtime_loaded_module": str(path),
                         "sha256": digest}:
            reasons.append(f"{label}: extension runtime identity differs")
    if fatal.get("qualified") is not True or any(
            (fatal.get("category_counts") or {}).values()):
        reasons.append(f"{label}: fatal scan differs")
    if (postflight.get("qualified") is not True
            or postflight.get("api_server_pids")
            or postflight.get("worker_pids")
            or postflight.get("gpu_processes")):
        reasons.append(f"{label}: postflight differs")
    return reasons, values


def reused_aa_reasons(args: argparse.Namespace,
                      arms: dict[str, dict[str, Any]],
                      historical: dict[str, Any]) -> list[str]:
    reference = arms["m1_109"]["runtime_manifest.json"]
    runtime = historical.get("runtime") or {}
    contract = historical.get("service_contract") or {}
    extensions = historical.get("extensions") or {}
    reasons = []
    if (historical.get("instance") != args.instance
            or historical.get("model_path") != str(reference.get("model_path"))
            or runtime.get("identity") != reference.get("runtime_identity")
            or contract.get("tensor_parallel_size") != 4
            or contract.get("dtype") != "float16"
            or contract.get("max_model_len") != 262144
            or contract.get("block_size") != 16
            or contract.get("targets") != [4096, 16384, 32768, 65536]
            or contract.get("positions_per_request") != 64
            or extensions.get("m1_109_fp32_qk", {}).get("sha256")
            != args.m1_109_sha256):
        reasons.append("M1-179 A/A runtime/request/extension identity differs")
    return reasons


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
    parser.add_argument("--m1-179-summary", type=Path, required=True)
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.m1_109_extension = args.m1_109_extension.resolve()
    args.m1_162_extension = args.m1_162_extension.resolve()
    if args.run_root.exists() or not args.run_root.is_absolute():
        parser.error("run root must be a new absolute path")
    for path in (args.session_preflight, args.m1_109_extension,
                 args.m1_162_extension, args.m1_179_summary):
        if not path.is_file():
            parser.error(f"required input missing: {path}")
    args.run_root.mkdir(parents=True)
    (args.run_root / "runtime-workdir").mkdir()
    started = time.monotonic()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{ROOT / 'tests'}:{ROOT / 'scripts'}:"
        f"{environment.get('PYTHONPATH', '')}")
    environment["BI100_TEACHER_FORCED_HMAC_KEY"] = secrets.token_hex(32)
    arms: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    try:
        for label in ARM_CONFIG:
            print(json.dumps({"event": "arm_start", "arm": label}), flush=True)
            rc = subprocess.run(
                arm_command(args, label), check=False, env=environment,
                cwd=args.run_root / "runtime-workdir").returncode
            arm_reasons, evidence = validate_arm(args, label)
            if evidence:
                arms[label] = evidence
            if rc or arm_reasons:
                reasons.extend(arm_reasons or [f"{label}: runner rc={rc}"])
                break
            print(json.dumps({"event": "arm_end", "arm": label,
                              "returncode": rc}), flush=True)
        final_postflight_path = args.run_root / "final_postflight.json"
        postflight_rc = subprocess.run([
            sys.executable, str(ROOT / "tests/service_postflight_gate.py"),
            "--gpus", "0,1,2,3", "--out", str(final_postflight_path),
        ], check=False, env=environment,
            cwd=args.run_root / "runtime-workdir").returncode
        final_postflight = (_load(final_postflight_path)
                            if final_postflight_path.is_file() else {})
        if postflight_rc or final_postflight.get("qualified") is not True:
            reasons.append("final service/GPU postflight failed")
        historical = _load(args.m1_179_summary)
        if not reasons and len(arms) == 3:
            reasons.extend(comparison.cross_arm_reasons({
                label: evidence["measurement.json"]
                for label, evidence in arms.items()}))
            reasons.extend(reused_aa_reasons(args, arms, historical))
        if reasons or len(arms) != 3:
            result = {"schema": SCHEMA, "version": 1, "status": "invalid",
                      "classification": "invalid_evidence", "reasons": reasons,
                      "promotion_authorized": False}
            rc = 2
        else:
            result = comparison.compare(
                arms["fused_off"]["measurement.json"],
                arms["m1_109"]["measurement.json"],
                arms["m1_162"]["measurement.json"],
                historical["aa_distribution"])
            rc = {"inconclusive": 3, "fail": 1, "invalid": 2}[result["status"]]
        result["experiment"] = {
            "schema": SCHEMA, "version": 1,
            "pair_id": args.pair_id, "arm_order": list(ARM_CONFIG),
            "service_startups": len(arms),
            "wall_s": time.monotonic() - started,
            "session_preflight": str(args.session_preflight),
            "all_arm_postflights_qualified": all(
                value["postflight_after.json"].get("qualified") is True
                for value in arms.values()),
            "fatal_category_counts": {
                label: value["fatal_scan.json"].get("category_counts", {})
                for label, value in arms.items()},
            "final_postflight": final_postflight,
        }
        _write(args.run_root / "summary.json", result)
        return rc
    finally:
        environment.pop("BI100_TEACHER_FORCED_HMAC_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
