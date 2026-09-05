#!/usr/bin/env python3
"""Run frozen IFEval-64 plus focused teacher-forced requests for M1-181."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import ifeval_quality_api as ifeval
import quality_gate_api as quality
import teacher_forced_topk_api as teacher


SCHEMA = "bi100-m1-181-arm-observation-v1"
SMOKE_ORDINALS = tuple(range(0, 64, 4))
SMOKE_BASELINE_ONLY_STOP = 3
TF_TARGETS = (4096, 16384, 32768, 65536)


def _atomic_json(path: Path, value: dict[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
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


def _case(row: dict[str, Any], response: dict[str, Any],
          score: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": row["key"],
        "instruction_id_list": list(row["instruction_id_list"]),
        "strict": list(score["strict"]),
        "loose": list(score["loose"]),
        "http_status": 200,
        "finish_reason": response["finish_reason"],
        "usage": response["usage"],
        "elapsed_s": response["elapsed_s"],
        "all_values_finite": True,
    }


def _score(rows: list[dict[str, Any]],
           responses: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    scores = ifeval.score_rows(
        rows, {key: value["content"] for key, value in responses.items()})
    by_key = {item["key"]: item for item in scores}
    return [_case(row, responses[row["key"]], by_key[row["key"]])
            for row in rows]


def smoke_baseline_only(baseline: dict[str, Any],
                        candidate: list[dict[str, Any]]) -> dict[str, int]:
    left = {item["key"]: item for item in baseline.get("ifeval", {}).get(
        "cases", [])}
    strict = loose = 0
    for right in candidate:
        lhs = left.get(right["key"])
        if lhs is None:
            raise ValueError("smoke baseline key is missing")
        strict += int(all(lhs["strict"]) and not all(right["strict"]))
        loose += int(all(lhs["loose"]) and not all(right["loose"]))
    return {"strict": strict, "loose": loose}


def run_ifeval(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    environment_root = Path(os.environ.get("BI100_IFEVAL_ENV", ""))
    site = environment_root / "site-packages"
    nltk_data = environment_root / "nltk_data"
    if not site.is_dir() or not nltk_data.is_dir():
        raise ValueError("offline IFEval environment is missing")
    sys.path.insert(0, str(site))
    os.environ["NLTK_DATA"] = str(nltk_data)
    manifest, manifest_sha, rows = ifeval.load_manifest(args.manifest)
    if len(rows) != 64:
        raise ValueError("M1-181 requires the frozen IFEval-64 subset")
    client = quality.Client(args.base)
    client.models("llm")
    checkpoint = args.out.with_name("ifeval_private_checkpoint.json")
    responses: dict[int, dict[str, Any]] = {}
    request_order = (list(SMOKE_ORDINALS)
                     + [index for index in range(64)
                        if index not in set(SMOKE_ORDINALS)]
                     if args.arm == "m1_109" else list(range(64)))
    smoke_counts = {"strict": 0, "loose": 0}
    stopped = False
    for request_number, index in enumerate(request_order, 1):
        row = rows[index]
        body, elapsed = ifeval.post(
            args.base, ifeval.request_payload(row["prompt"], "llm", manifest),
            args.timeout_s)
        normalized = ifeval.normalize_response(body, elapsed)
        responses[row["key"]] = normalized
        _atomic_json(checkpoint, {
            "schema": "bi100-m1-181-private-ifeval-checkpoint-v1",
            "version": 1, "arm": args.arm,
            "completed_keys": list(responses), "responses": responses,
            "contains_raw_model_outputs": True,
            "must_remain_outside_repository": True,
        })
        if args.arm == "m1_109" and request_number == len(SMOKE_ORDINALS):
            smoke_rows = [rows[i] for i in SMOKE_ORDINALS]
            smoke_cases = _score(smoke_rows, responses)
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            smoke_counts = smoke_baseline_only(baseline, smoke_cases)
            stopped = max(smoke_counts.values()) >= SMOKE_BASELINE_ONLY_STOP
            if stopped:
                break
    completed_rows = [row for row in rows if row["key"] in responses]
    cases = _score(completed_rows, responses)
    checkpoint.unlink(missing_ok=True)
    return ({
        "manifest_name": args.manifest.name,
        "manifest_sha256": manifest_sha,
        "evaluator_revision": manifest["evaluator"]["revision"],
        "request_conversion": manifest["request_conversion"],
        "smoke_ordinals": list(SMOKE_ORDINALS),
        "smoke_baseline_only_stop": SMOKE_BASELINE_ONLY_STOP,
        "smoke_baseline_only": smoke_counts,
        "stopped_after_smoke": stopped,
        "cases": sorted(cases, key=lambda item: rows.index(next(
            row for row in rows if row["key"] == item["key"]))),
        "selected": 64, "completed": len(cases),
        "complete": len(cases) == 64,
        "checkpoint_deleted": True,
    }, len(cases))


def run_teacher(args: argparse.Namespace) -> dict[str, Any]:
    output = args.out.with_name("teacher_forced_private.json")
    command = [
        sys.executable, str(Path(teacher.__file__)),
        "--base", args.base, "--model-path", str(args.model_path),
        "--attention-runtime-manifest", str(args.attention_runtime_manifest),
        "--runtime-identity", args.runtime_identity,
        "--source-revision", args.source_revision,
        "--instance", args.instance,
        "--mode", "candidate" if args.arm == "m1_109" else "control",
        "--targets", ",".join(map(str, TF_TARGETS)),
        "--timeout-s", "3600", "--out", str(output),
    ]
    completed = subprocess.run(command, check=False, env=os.environ.copy())
    if completed.returncode or not output.is_file():
        raise RuntimeError("teacher-forced workload failed")
    value = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("teacher-forced observation is not an object")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    ifeval_result = None
    count = 0
    if args.arm != "fused_off_b":
        ifeval_result, count = run_ifeval(args)
    teacher_forced = None
    if ifeval_result is None or not ifeval_result["stopped_after_smoke"]:
        teacher_forced = run_teacher(args)
        count += len(TF_TARGETS)
    return {
        "schema": SCHEMA, "version": 1, "arm": args.arm,
        "algorithm_variant": ("m1_109_fp32_qk" if args.arm == "m1_109"
                              else "fused_off"),
        "source_revision": args.source_revision,
        "runtime_identity": args.runtime_identity,
        "instance": args.instance, "model_path": str(args.model_path),
        "workload_id": args.workload_id,
        "ifeval_environment": os.environ.get("BI100_IFEVAL_ENV"),
        "ifeval": ifeval_result, "teacher_forced": teacher_forced,
        "request_population": {"attempted": count, "completed": count,
                               "failed": 0},
        "wall_s": time.monotonic() - started,
        "privacy": {"contains_prompts": False,
                    "contains_model_outputs": False,
                    "contains_token_ids_or_identity_key": False,
                    "contains_credentials": False,
                    "private_checkpoints_outside_repository": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--attention-runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--arm", choices=("fused_off", "m1_109", "fused_off_b"),
                        required=True)
    parser.add_argument("--manifest", type=Path, default=ifeval.DEFAULT_MANIFEST)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.arm == "m1_109" and (args.baseline is None
                                  or not args.baseline.is_file()):
        parser.error("M1-109 requires the completed fused-off baseline")
    try:
        result = run(args)
        _atomic_json(args.out, result)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"M1-181 workload invalid: {type(exc).__name__}: {exc}",
              file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
