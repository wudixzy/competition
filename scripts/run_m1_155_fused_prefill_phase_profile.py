#!/usr/bin/env python3
"""Run candidate-only ixprof phase attribution on three healthy BI100s."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time
from typing import Any

import run_m1_142_l1_subset_screen as lifecycle


ROOT = Path(__file__).resolve().parents[1]
CELL_SCRIPT = ROOT / "tests" / "profile_m1_155_fused_prefill_phase.py"
QUALIFIER = ROOT / "tests" / "qualify_m1_155_ixprof_phase.py"
IXPROF = Path("/usr/local/corex-3.2.3/bin/ixprof")
CASES = (
    "p90_total_16k_q8176",
    "p90_total_32k_q8176",
    "p90_total_64k_q8176",
)
RUNNER_SCHEMA = "bi100-m1-155-fused-prefill-phase-runner-v1"


def _spawn_cell(
    *,
    case: str,
    gpu: int,
    extension: Path,
    extension_sha: str,
    revision: str,
    instance: str,
    run_root: Path,
) -> lifecycle.Child:
    environment = lifecycle._base_environment()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    stdout = (run_root / f"{case}.stdout").open("wb")
    stderr = (run_root / f"{case}.stderr").open("wb")
    try:
        return lifecycle._spawn_managed(
            label=case,
            gpu=gpu,
            command=[
                "timeout", "--foreground", "--signal=TERM",
                "--kill-after=90s", "1800s",
                str(IXPROF),
                "--profile-from-start", "off",
                "--print-gpu-summary",
                sys.executable,
                str(CELL_SCRIPT),
                "--case", case,
                "--extension", str(extension),
                "--expected-extension-sha256", extension_sha,
                "--source-revision", revision,
                "--instance", instance,
                "--visible-physical-gpu", str(gpu),
                "--output", str(run_root / f"{case}.json"),
            ],
            stdout=stdout,
            stderr=stderr,
            environment=environment,
        )
    finally:
        stdout.close()
        stderr.close()


def _run_cells(
    *,
    gpus: list[int],
    extension: Path,
    extension_sha: str,
    revision: str,
    instance: str,
    run_root: Path,
) -> list[dict[str, Any]]:
    children = [
        _spawn_cell(
            case=case,
            gpu=gpu,
            extension=extension,
            extension_sha=extension_sha,
            revision=revision,
            instance=instance,
            run_root=run_root,
        )
        for case, gpu in zip(CASES, gpus)
    ]
    while any(child.process.poll() is None for child in children):
        time.sleep(0.1)
    rows = []
    for child in children:
        rows.append({
            "case": child.case,
            "gpu": child.gpu,
            "returncode": child.process.wait(),
            "elapsed_s": time.monotonic() - child.started,
        })
        lifecycle._ACTIVE_CHILDREN.remove(child)
    return rows


def _qualify_cells(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for case in CASES:
        output = run_root / f"{case}.phase.json"
        rc = lifecycle._run_to_files(
            [
                sys.executable,
                str(QUALIFIER),
                "--cell", str(run_root / f"{case}.json"),
                "--profile-log", str(run_root / f"{case}.stdout"),
                "--profile-log", str(run_root / f"{case}.stderr"),
                "--output", str(output),
            ],
            run_root / f"{case}.qualify.stdout",
            run_root / f"{case}.qualify.stderr",
            label=f"{case}_qualify",
            timeout_s=60,
            environment=lifecycle._base_environment(),
        )
        report = (
            json.loads(output.read_text(encoding="ascii"))
            if output.is_file() else {}
        )
        rows.append({
            "case": case,
            "returncode": rc,
            "qualified": report.get("qualified") is True,
            "report": report,
        })
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    observed = {row["case"] for row in rows}
    if observed != set(CASES):
        reasons.append("fixed profile cases are incomplete")
    for row in rows:
        if row["returncode"] != 0 or not row["qualified"]:
            reasons.append(f"{row['case']}: phase attribution failed")
    qualified = not reasons
    return {
        "schema": "bi100-m1-155-fused-prefill-phase-screen-v1",
        "version": 1,
        "qualified": qualified,
        "reasons": reasons,
        "rows": [row["report"] for row in rows],
        "authorization": {
            "implementation_direction_authorized": qualified,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }


def run(args: argparse.Namespace) -> int:
    extension, extension_sha, revision = lifecycle._validate(args)
    if (
        not IXPROF.is_file()
        or not extension.is_relative_to(Path("/tmp"))
        or extension.stat().st_mode & 0o022
    ):
        raise ValueError(
            "M1-155 requires ixprof and a private extension under /tmp")
    if len(args.gpus) != len(CASES):
        raise ValueError("M1-155 requires exactly three physical GPUs")
    run_root = args.run_root.resolve()
    run_root.mkdir(mode=0o700, parents=True)
    started = time.monotonic()
    stage = "initialization"
    before_preflight = False
    cell_rows: list[dict[str, Any]] = []
    screen: dict[str, Any] = {
        "qualified": False,
        "reasons": ["phase profile did not run"],
        "rows": [],
        "authorization": {
            "implementation_direction_authorized": False,
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }
    primary_ok = True
    cleanup_ok = True
    postflight_ok = False
    after_preflight_ok = False
    comparison_ok = False
    try:
        lifecycle._atomic_json(run_root / "identity.json", {
            "source_revision": revision,
            "source_branch": lifecycle._git("branch", "--show-current"),
            "instance": args.instance,
            "gpus": args.gpus,
            "extension_sha256": extension_sha,
            "kernel_source_sha256": lifecycle._sha256(
                ROOT / "qwen3_6_scripts"
                / "corex_fused_paged_prefill_split4.cu"),
            "ixprof_path": str(IXPROF),
        })
        stage = "postflight_before"
        if lifecycle._run_postflight(run_root, stage, args.gpus) != 0:
            raise RuntimeError("initial postflight failed")
        stage = "preflight_before"
        if lifecycle._run_preflight(run_root, stage, args.gpus) != 0:
            raise RuntimeError("initial preflight failed")
        before_preflight = True
        stage = "candidate_profile"
        cell_rows = _run_cells(
            gpus=args.gpus,
            extension=extension,
            extension_sha=extension_sha,
            revision=revision,
            instance=args.instance,
            run_root=run_root,
        )
        if any(row["returncode"] != 0 for row in cell_rows):
            raise RuntimeError("one or more ixprof cells failed")
        stage = "phase_qualification"
        qualified_rows = _qualify_cells(run_root)
        screen = _aggregate(qualified_rows)
        lifecycle._atomic_json(run_root / "screen.json", screen)
        if not screen["qualified"]:
            raise RuntimeError("phase attribution did not qualify")
    except lifecycle.ParentTermination:
        primary_ok = False
        raise
    except BaseException as exc:  # noqa: BLE001
        primary_ok = False
        lifecycle._atomic_json(run_root / "failure.json", {
            "stage": stage,
            "error_type": type(exc).__name__,
            "message_recorded": False,
        })
    finally:
        cleanup_ok = lifecycle.cleanup_children(
            lifecycle._ACTIVE_CHILDREN)
        lifecycle._ACTIVE_CHILDREN.clear()
        postflight_ok = (
            lifecycle._run_postflight(
                run_root, "postflight_after", args.gpus) == 0)
        if before_preflight and cleanup_ok and postflight_ok:
            after_preflight_ok = (
                lifecycle._run_preflight(
                    run_root, "preflight_after", args.gpus) == 0)
        if after_preflight_ok:
            comparison_ok = lifecycle._run_to_files(
                [
                    sys.executable,
                    str(ROOT / "tests"
                        / "compare_bi100_preflights.py"),
                    "--preflight",
                    f"before={run_root / 'preflight_before.json'}",
                    "--preflight",
                    f"after={run_root / 'preflight_after.json'}",
                    "--expected-gpus", ",".join(map(str, args.gpus)),
                    "--max-free-memory-drop-bytes", "1073741824",
                    "--out", str(run_root / "preflight_comparison.json"),
                ],
                run_root / "preflight_comparison.stdout",
                run_root / "preflight_comparison.stderr",
                label="preflight_comparison",
                timeout_s=300,
                environment=lifecycle._base_environment(),
            ) == 0
        fatal = lifecycle._scan_fatal(run_root)
        lifecycle._atomic_json(run_root / "fatal_scan.json", fatal)
        source_unchanged = (
            lifecycle._git("rev-parse", "HEAD") == revision
            and not lifecycle._git(
                "status", "--porcelain", "--untracked-files=all",
                "--", ".", ":(exclude)bench_runs/**")
        )
        qualified = all((
            primary_ok,
            screen.get("qualified") is True,
            cleanup_ok,
            postflight_ok,
            after_preflight_ok,
            comparison_ok,
            fatal.get("qualified") is True,
            source_unchanged,
        ))
        lifecycle._atomic_json(run_root / "runner_status.json", {
            "schema": RUNNER_SCHEMA,
            "version": 1,
            "qualified": qualified,
            "terminal_stage": stage,
            "source_revision": revision,
            "instance": args.instance,
            "gpus": args.gpus,
            "gpu_count": len(args.gpus),
            "fixed_cases": list(CASES),
            "wall_s": time.monotonic() - started,
            "extension_sha256": extension_sha,
            "cell_processes": cell_rows,
            "screen": screen,
            "lifecycle": {
                "cleanup_reaped": cleanup_ok,
                "fatal_scan_qualified": fatal.get("qualified"),
                "postflight_qualified": postflight_ok,
                "after_preflight_qualified": after_preflight_ok,
                "preflight_comparison_qualified": comparison_ok,
                "source_unchanged": source_unchanged,
            },
            "authorization": screen.get("authorization"),
            "privacy": {
                "raw_tensors_recorded": False,
                "model_outputs_recorded": False,
                "prompts_recorded": False,
                "credentials_recorded": False,
            },
        })
    return 0 if json.loads(
        (run_root / "runner_status.json").read_text(
            encoding="ascii"))["qualified"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("extension", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--gpus",
        type=lifecycle.parse_gpus,
        default=lifecycle.parse_gpus("1,2,3"),
    )
    args = parser.parse_args()
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in previous_handlers:
        signal.signal(signum, lifecycle._signal_handler)
    try:
        return run(args)
    except lifecycle.ParentTermination as termination:
        lifecycle.cleanup_children(lifecycle._ACTIVE_CHILDREN)
        return 128 + termination.signum
    finally:
        lifecycle.cleanup_children(lifecycle._ACTIVE_CHILDREN)
        lifecycle._ACTIVE_CHILDREN.clear()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
