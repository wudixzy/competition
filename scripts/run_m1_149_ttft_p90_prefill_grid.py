#!/usr/bin/env python3
"""Run the M1-149 medium-context operator grid on healthy BI100 cards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import statistics
import sys
import time
from typing import Any

import run_m1_142_l1_subset_screen as lifecycle


ROOT = Path(__file__).resolve().parents[1]
CELL_SCRIPT = ROOT / "tests" / "bench_m1_149_ttft_p90_prefill.py"
CELL_SCHEMA = "bi100-m1-149-ttft-p90-prefill-cell-v1"
RUNNER_SCHEMA = "bi100-m1-149-ttft-p90-prefill-grid-v1"
MIN_SPEEDUP = 1.2
CASES = tuple(
    f"p90_total_{total_k // 1024:02d}k_q8176"
    for total_k in range(8192, 65537, 8192)
)


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
                "--kill-after=90s", "3600s",
                sys.executable,
                str(CELL_SCRIPT),
                "--case", case,
                "--extension", str(extension),
                "--expected-extension-sha256", extension_sha,
                "--source-commit", revision,
                "--runtime-identity", "corex-3.2.3-m1-149",
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


def _run_waves(
    *,
    gpus: list[int],
    extension: Path,
    extension_sha: str,
    revision: str,
    instance: str,
    run_root: Path,
) -> list[dict[str, Any]]:
    assignments = [
        (case, gpus[index % len(gpus)])
        for index, case in enumerate(CASES)
    ]
    rows = []
    for offset in range(0, len(assignments), len(gpus)):
        wave = assignments[offset:offset + len(gpus)]
        wave_started = time.monotonic()
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
            for case, gpu in wave
        ]
        while any(child.process.poll() is None for child in children):
            time.sleep(0.1)
        wave_rows = []
        for child in children:
            row = {
                "case": child.case,
                "gpu": child.gpu,
                "returncode": child.process.wait(),
                "elapsed_s": time.monotonic() - child.started,
            }
            rows.append(row)
            wave_rows.append(row)
            lifecycle._ACTIVE_CHILDREN.remove(child)
        lifecycle._atomic_json(
            run_root / f"wave-{offset // len(gpus)}.json",
            {
                "wave": offset // len(gpus),
                "wall_s": time.monotonic() - wave_started,
                "cells": wave_rows,
            },
        )
        if any(row["returncode"] != 0 for row in wave_rows):
            break
    return rows


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def aggregate(
    reports: list[dict[str, Any]],
    *,
    gpus: list[int],
    extension_sha: str,
    source_revision: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    rows = []
    observed = set()
    for report in reports:
        case = report.get("case")
        if case not in CASES or case in observed:
            reasons.append(f"invalid or duplicate case: {case!r}")
            continue
        observed.add(case)
        numerical = report.get("numerical") or {}
        timings = report.get("timings") or {}
        evaluation = report.get("evaluation") or {}
        extension = report.get("extension") or {}
        gpu = report.get("visible_physical_gpu")
        speedup = timings.get("speedup")
        row = {
            "case": case,
            "total_kv_len": report.get("total_kv_len"),
            "gpu": gpu,
            "speedup": speedup if _finite(speedup) else None,
            "reference_cuda_median_ms": (
                (timings.get("reference") or {}).get("cuda_median_ms")),
            "candidate_cuda_median_ms": (
                (timings.get("candidate") or {}).get("cuda_median_ms")),
            "output_relative_l2": numerical.get("output_relative_l2"),
            "lse_relative_l2": numerical.get("lse_relative_l2"),
            "output_max_abs": numerical.get("output_max_abs"),
            "finite": numerical.get("finite"),
            "qualified": evaluation.get("qualified") is True,
        }
        rows.append(row)
        expected_gpu = gpus[CASES.index(case) % len(gpus)]
        if report.get("schema") != CELL_SCHEMA:
            reasons.append(f"{case}: cell schema differs")
        if report.get("source_commit") != source_revision:
            reasons.append(f"{case}: source revision differs")
        if gpu != expected_gpu:
            reasons.append(f"{case}: physical GPU assignment differs")
        if extension.get("sha256") != extension_sha:
            reasons.append(f"{case}: extension identity differs")
        if evaluation.get("qualified") is not True:
            reasons.append(f"{case}: cell evaluation failed")
        if not _finite(speedup) or float(speedup) < MIN_SPEEDUP:
            reasons.append(f"{case}: speedup is below {MIN_SPEEDUP:.1f}x")
    missing = [case for case in CASES if case not in observed]
    if missing:
        reasons.append(f"missing cases: {missing}")
    rows.sort(key=lambda row: CASES.index(row["case"]))
    speedups = [
        float(row["speedup"]) for row in rows if _finite(row["speedup"])
    ]
    cumulative_rows = []
    cumulative_reference = 0.0
    cumulative_candidate = 0.0
    for row in rows:
        reference_ms = row["reference_cuda_median_ms"]
        candidate_ms = row["candidate_cuda_median_ms"]
        if (
            not _finite(reference_ms)
            or not _finite(candidate_ms)
            or float(reference_ms) <= 0.0
            or float(candidate_ms) <= 0.0
        ):
            reasons.append(f"{row['case']}: timing median is invalid")
            continue
        cumulative_reference += float(reference_ms)
        cumulative_candidate += float(candidate_ms)
        cumulative_rows.append({
            "total_prompt_tokens": row["total_kv_len"],
            "reference_attention_ms": cumulative_reference,
            "candidate_attention_ms": cumulative_candidate,
            "candidate_speedup": (
                cumulative_reference / cumulative_candidate),
            "candidate_reduction_fraction": (
                1.0 - cumulative_candidate / cumulative_reference),
        })
    qualified = not reasons
    return {
        "schema": "bi100-m1-149-ttft-p90-prefill-screen-v1",
        "version": 1,
        "qualified": qualified,
        "reasons": reasons,
        "rows": rows,
        "cumulative_attention_only_estimate": cumulative_rows,
        "minimum_speedup": min(speedups) if speedups else None,
        "median_speedup": statistics.median(speedups) if speedups else None,
        "authorization": {
            "short_tp4_p90_screen_authorized": qualified,
            "l2_capture_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }


def _load_reports(run_root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="ascii"))
        for case in CASES
        if (path := run_root / f"{case}.json").is_file()
    ]


def run(args: argparse.Namespace) -> int:
    extension, extension_sha, revision = lifecycle._validate(args)
    if (
        not extension.is_relative_to(Path("/tmp"))
        or extension.stat().st_mode & 0o022
    ):
        raise ValueError(
            "M1-149 extension must be private and not group/other writable "
            "under /tmp")
    run_root = args.run_root.resolve()
    run_root.mkdir(mode=0o700, parents=True)
    started = time.monotonic()
    stage = "initialization"
    before_preflight = False
    cell_rows: list[dict[str, Any]] = []
    primary_ok = True
    cleanup_ok = True
    postflight_ok = False
    after_preflight_ok = False
    comparison_ok = False
    screen: dict[str, Any] = {
        "qualified": False,
        "reasons": ["P90 grid did not run"],
        "rows": [],
        "authorization": {
            "short_tp4_p90_screen_authorized": False,
            "l2_capture_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }
    try:
        lifecycle._atomic_json(run_root / "identity.json", {
            "source_revision": revision,
            "source_branch": lifecycle._git(
                "branch", "--show-current"),
            "instance": args.instance,
            "gpus": args.gpus,
            "extension_sha256": extension_sha,
            "kernel_source_sha256": lifecycle._sha256(
                ROOT / "qwen3_6_scripts"
                / "corex_fused_paged_prefill_split4.cu"),
        })
        stage = "postflight_before"
        if lifecycle._run_postflight(
                run_root, stage, args.gpus) != 0:
            raise RuntimeError("initial postflight failed")
        stage = "preflight_before"
        if lifecycle._run_preflight(
                run_root, stage, args.gpus) != 0:
            raise RuntimeError("initial preflight failed")
        before_preflight = True
        stage = "operator_waves"
        cell_rows = _run_waves(
            gpus=args.gpus,
            extension=extension,
            extension_sha=extension_sha,
            revision=revision,
            instance=args.instance,
            run_root=run_root,
        )
        if (
            len(cell_rows) != len(CASES)
            or any(row["returncode"] != 0 for row in cell_rows)
        ):
            raise RuntimeError("one or more P90 cells failed")
        screen = aggregate(
            _load_reports(run_root),
            gpus=args.gpus,
            extension_sha=extension_sha,
            source_revision=revision,
        )
        lifecycle._atomic_json(run_root / "screen.json", screen)
        if not screen["qualified"]:
            raise RuntimeError("P90 operator screen did not qualify")
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
                    "--expected-gpus",
                    ",".join(map(str, args.gpus)),
                    "--max-free-memory-drop-bytes", "1073741824",
                    "--out", str(
                        run_root / "preflight_comparison.json"),
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
            "waves": math.ceil(len(CASES) / len(args.gpus)),
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
                "prompts_recorded": False,
                "model_outputs_recorded": False,
                "token_ids_recorded": False,
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
