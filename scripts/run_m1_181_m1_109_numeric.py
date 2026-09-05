#!/usr/bin/env python3
"""Run the fixed two-cell M1-109 real-activation replay on four GPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "tests/replay_m1_181_m1_109_activation.py"
SCHEMA = "bi100-m1-181-m1-109-numeric-v1"


def commands(args: argparse.Namespace) -> list[list[str]]:
    return [[
        sys.executable, str(REPLAY),
        "--bank-manifest", str(args.bank_root / f"rank-{rank}-bank"
                               / f"logical-rank-{rank}.manifest.json"),
        "--extension", str(args.extension),
        "--extension-sha256", args.extension_sha256,
        "--logical-rank", str(rank), "--physical-gpu", str(rank),
        "--out", str(args.run_root / f"rank-{rank}.json"),
    ] for rank in range(4)]


def validate(args: argparse.Namespace) -> None:
    if args.run_root.exists() or not args.run_root.is_absolute():
        raise ValueError("run root must be a new absolute path")
    if not args.extension.is_file():
        raise ValueError("M1-109 extension is missing")
    if (len(args.extension_sha256) != 64
            or any(char not in "0123456789abcdef"
                   for char in args.extension_sha256)):
        raise ValueError("extension SHA-256 identity is invalid")
    for command in commands(args):
        manifest = Path(command[command.index("--bank-manifest") + 1])
        if not manifest.is_file():
            raise ValueError(f"activation bank missing: {manifest}")


def aggregate(run_root: Path, returncodes: list[int], wall_s: float) -> dict[str, Any]:
    reasons = []
    ranks = []
    for rank, returncode in enumerate(returncodes):
        path = run_root / f"rank-{rank}.json"
        if returncode or not path.is_file():
            reasons.append(f"rank {rank}: replay returncode {returncode}")
            continue
        value = json.loads(path.read_text(encoding="ascii"))
        records = value.get("records")
        if (value.get("schema") != "bi100-m1-181-m1-109-rank-replay-v1"
                or value.get("logical_tp_rank") != rank
                or value.get("physical_gpu") != rank
                or value.get("all_qualified") is not True
                or not isinstance(records, list) or len(records) != 2):
            reasons.append(f"rank {rank}: replay evidence differs")
            continue
        ranks.append(value)
    status = "pass" if not reasons and len(ranks) == 4 else (
        "fail" if len(ranks) == 4 else "invalid")
    numerics = [record["numeric"] for rank in ranks
                for record in rank["records"]]
    return {
        "schema": SCHEMA, "version": 1, "status": status,
        "classification": ("m1_109_real_activation_g2_pass" if status == "pass"
                           else "m1_109_real_activation_g2_failed"
                           if status == "fail" else "invalid_evidence"),
        "reasons": reasons,
        "rank_count": len(ranks),
        "cell_count": len(numerics),
        "all_finite": bool(numerics) and all(n["all_finite"] for n in numerics),
        "all_repeats_exact": bool(ranks) and all(
            all(all(record["repeat_exact"].values())
                for record in rank["records"]) for rank in ranks),
        "maximum_relative_l2_error_ratio": (
            max(n["relative_l2_error_ratio"] for n in numerics)
            if numerics else None),
        "maximum_absolute_error_ratio": (
            max(n["maximum_absolute_error_ratio"] for n in numerics)
            if numerics else None),
        "maximum_candidate_lse_relative_l2": (
            max(n["candidate_lse_relative_l2"] for n in numerics)
            if numerics else None),
        "rank_summaries": [{
            "logical_tp_rank": rank["logical_tp_rank"],
            "physical_gpu": rank["physical_gpu"],
            "wall_s": rank["wall_s"],
            "records": rank["records"],
        } for rank in ranks],
        "wall_s": wall_s,
        "operator_screen_only": True,
        "full_model_tp4_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.bank_root = args.bank_root.resolve()
    args.extension = args.extension.resolve()
    validate(args)
    plan = {"schema": "bi100-m1-181-numeric-plan-v1", "version": 1,
            "commands": commands(args), "timeout_s": args.timeout_s}
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    args.run_root.mkdir(parents=True)
    (args.run_root / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.monotonic()
    children = []
    for rank, command in enumerate(commands(args)):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(rank)
        stdout = (args.run_root / f"rank-{rank}.stdout").open("wb")
        stderr = (args.run_root / f"rank-{rank}.stderr").open("wb")
        process = subprocess.Popen(command, env=environment,
                                   stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        stdout.close()
        stderr.close()
        children.append(process)
    deadline = started + args.timeout_s
    timed_out = False
    try:
        while any(child.poll() is None for child in children):
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(1)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        term_deadline = time.monotonic() + 45
        while (any(child.poll() is None for child in children)
               and time.monotonic() < term_deadline):
            time.sleep(1)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
        returncodes = [child.wait() for child in children]
    result = aggregate(args.run_root, returncodes, time.monotonic() - started)
    if timed_out:
        result.update({"status": "invalid", "classification": "timeout",
                       "reasons": ["fixed replay exceeded wall-time budget"]})
    (args.run_root / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return {"pass": 0, "fail": 1, "invalid": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
