#!/usr/bin/env python3
"""Run the fixed M1-157 QK A/B screen on three healthy BI100 cards."""

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
CELL_SCRIPT = ROOT / "tests" / "bench_m1_157_fp16_qk_ab.py"
CASES = (
    "p90_total_16k_q8176",
    "p90_total_32k_q8176",
    "p90_total_64k_q8176",
)
RUNNER_SCHEMA = "bi100-m1-157-fp16-qk-ab-runner-v1"
SCREEN_SCHEMA = "bi100-m1-157-fp16-qk-ab-screen-v1"
RUNTIME_IDENTITY = "corex-3.2.3-m1-157"
MIN_MEDIAN_SPEEDUP = 1.08
BASELINE_MODULE_NAME = "corex_fused_paged_prefill"


def screen_authorization(qualified: bool) -> dict[str, bool]:
    return {
        "short_tp4_screen_authorized": qualified,
        "long_context_or_quality_authorized": False,
        "main_or_yaml_change_authorized": False,
    }


def _validate_artifact(path: Path) -> tuple[Path, str]:
    resolved = path.resolve(strict=True)
    if (
        not resolved.is_relative_to(Path("/tmp"))
        or resolved.stat().st_mode & 0o022
    ):
        raise ValueError(
            "M1-157 artifacts must be private and not writable by "
            "group/other under /tmp"
        )
    return resolved, lifecycle._sha256(resolved)


def _spawn_cell(
    *,
    case: str,
    gpu: int,
    baseline: Path,
    baseline_sha: str,
    baseline_module_name: str,
    candidate: Path,
    candidate_sha: str,
    candidate_module_name: str,
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
                "timeout",
                "--foreground",
                "--signal=TERM",
                "--kill-after=90s",
                "1800s",
                sys.executable,
                str(CELL_SCRIPT),
                "--case",
                case,
                "--baseline-extension",
                str(baseline),
                "--candidate-extension",
                str(candidate),
                "--expected-baseline-sha256",
                baseline_sha,
                "--baseline-module-name",
                baseline_module_name,
                "--expected-candidate-sha256",
                candidate_sha,
                "--candidate-module-name",
                candidate_module_name,
                "--source-revision",
                revision,
                "--runtime-identity",
                RUNTIME_IDENTITY,
                "--instance",
                instance,
                "--visible-physical-gpu",
                str(gpu),
                "--output",
                str(run_root / f"{case}.json"),
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
    baseline: Path,
    baseline_sha: str,
    baseline_module_name: str,
    candidate: Path,
    candidate_sha: str,
    candidate_module_name: str,
    revision: str,
    instance: str,
    run_root: Path,
) -> list[dict[str, Any]]:
    children = [
        _spawn_cell(
            case=case,
            gpu=gpu,
            baseline=baseline,
            baseline_sha=baseline_sha,
            baseline_module_name=baseline_module_name,
            candidate=candidate,
            candidate_sha=candidate_sha,
            candidate_module_name=candidate_module_name,
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
        rows.append(
            {
                "case": child.case,
                "gpu": child.gpu,
                "returncode": child.process.wait(),
                "elapsed_s": time.monotonic() - child.started,
            }
        )
        lifecycle._ACTIVE_CHILDREN.remove(child)
    return rows


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def aggregate(
    reports: list[dict[str, Any]],
    *,
    revision: str,
    baseline_sha: str,
    candidate_sha: str,
    baseline_module_name: str = BASELINE_MODULE_NAME,
    candidate_module_name: str = "corex_fused_paged_prefill_fp16_qk",
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
        speedup = (report.get("timings") or {}).get("speedup")
        row = {
            "case": case,
            "gpu": report.get("visible_physical_gpu"),
            "speedup": speedup,
            "baseline_cuda_median_ms": (
                (report.get("timings") or {})
                .get("baseline", {})
                .get("cuda_median_ms")
            ),
            "candidate_cuda_median_ms": (
                (report.get("timings") or {})
                .get("candidate", {})
                .get("cuda_median_ms")
            ),
            "candidate_reference_output_l2": (
                (report.get("numerical") or {})
                .get("candidate_vs_reference", {})
                .get("output_relative_l2")
            ),
            "candidate_reference_lse_l2": (
                (report.get("numerical") or {})
                .get("candidate_vs_reference", {})
                .get("lse_relative_l2")
            ),
            "candidate_baseline_output_l2": (
                (report.get("numerical") or {})
                .get("candidate_vs_baseline", {})
                .get("output_relative_l2")
            ),
            "qualified": (report.get("evaluation") or {}).get("qualified"),
        }
        rows.append(row)
        expected_gpu = [1, 2, 3][CASES.index(case)]
        if report.get("source_revision") != revision:
            reasons.append(f"{case}: source revision differs")
        if report.get("visible_physical_gpu") != expected_gpu:
            reasons.append(f"{case}: physical GPU assignment differs")
        if (
            (report.get("baseline_extension") or {}).get("sha256")
            != baseline_sha
        ):
            reasons.append(f"{case}: baseline identity differs")
        if (
            (report.get("baseline_extension") or {}).get("module_name")
            != baseline_module_name
        ):
            reasons.append(f"{case}: baseline module name differs")
        if (
            (report.get("candidate_extension") or {}).get("sha256")
            != candidate_sha
        ):
            reasons.append(f"{case}: candidate identity differs")
        if (
            (report.get("candidate_extension") or {}).get("module_name")
            != candidate_module_name
        ):
            reasons.append(f"{case}: candidate module name differs")
        if (report.get("evaluation") or {}).get("qualified") is not True:
            reasons.append(f"{case}: cell gate failed")
    missing = [case for case in CASES if case not in observed]
    if missing:
        reasons.append(f"missing cases: {missing}")
    rows.sort(
        key=lambda row: CASES.index(row["case"])
        if row["case"] in CASES
        else len(CASES)
    )
    speedups = [
        float(row["speedup"])
        for row in rows
        if _finite_positive(row["speedup"])
    ]
    median_speedup = statistics.median(speedups) if speedups else None
    if (
        median_speedup is None
        or median_speedup < MIN_MEDIAN_SPEEDUP
    ):
        reasons.append(
            f"median speedup is below {MIN_MEDIAN_SPEEDUP:.2f}x"
        )
    qualified = not reasons
    return {
        "schema": SCREEN_SCHEMA,
        "version": 1,
        "qualified": qualified,
        "reasons": reasons,
        "rows": rows,
        "minimum_speedup": min(speedups) if speedups else None,
        "median_speedup": median_speedup,
        "authorization": screen_authorization(qualified),
    }


def _load_reports(run_root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="ascii"))
        for case in CASES
        if (path := run_root / f"{case}.json").is_file()
    ]


def run(args: argparse.Namespace) -> int:
    if args.gpus != [1, 2, 3]:
        raise ValueError("M1-157 requires the fixed physical GPUs 1,2,3")
    baseline, baseline_sha = _validate_artifact(args.baseline_extension)
    candidate, candidate_sha = _validate_artifact(args.candidate_extension)
    candidate_source = args.candidate_source.resolve(strict=True)
    if not candidate_source.is_relative_to(ROOT):
        raise ValueError("candidate source must be inside the repository")
    revision = lifecycle._git("rev-parse", "HEAD")
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
        "reasons": ["operator screen did not run"],
        "authorization": screen_authorization(False),
    }
    try:
        lifecycle._atomic_json(
            run_root / "identity.json",
            {
                "source_revision": revision,
                "source_branch": lifecycle._git("branch", "--show-current"),
                "instance": args.instance,
                "gpus": args.gpus,
                "baseline_extension_sha256": baseline_sha,
                "baseline_module_name": args.baseline_module_name,
                "candidate_extension_sha256": candidate_sha,
                "candidate_module_name": args.candidate_module_name,
                "candidate_source": str(candidate_source.relative_to(ROOT)),
                "baseline_source_sha256": lifecycle._sha256(
                    ROOT
                    / "qwen3_6_scripts"
                    / "corex_fused_paged_prefill_split4.cu"
                ),
                "candidate_source_sha256": lifecycle._sha256(
                    candidate_source
                ),
            },
        )
        stage = "postflight_before"
        if lifecycle._run_postflight(run_root, stage, args.gpus) != 0:
            raise RuntimeError("initial postflight failed")
        stage = "preflight_before"
        if lifecycle._run_preflight(run_root, stage, args.gpus) != 0:
            raise RuntimeError("initial preflight failed")
        before_preflight = True
        stage = "paired_operator_cells"
        cell_rows = _run_cells(
            gpus=args.gpus,
            baseline=baseline,
            baseline_sha=baseline_sha,
            baseline_module_name=args.baseline_module_name,
            candidate=candidate,
            candidate_sha=candidate_sha,
            candidate_module_name=args.candidate_module_name,
            revision=revision,
            instance=args.instance,
            run_root=run_root,
        )
        screen = aggregate(
            _load_reports(run_root),
            revision=revision,
            baseline_sha=baseline_sha,
            candidate_sha=candidate_sha,
            baseline_module_name=args.baseline_module_name,
            candidate_module_name=args.candidate_module_name,
        )
        lifecycle._atomic_json(run_root / "screen.json", screen)
        if (
            any(row["returncode"] != 0 for row in cell_rows)
            or not screen["qualified"]
        ):
            raise RuntimeError("M1-157 paired operator screen failed")
    except lifecycle.ParentTermination:
        primary_ok = False
        raise
    except BaseException as exc:  # noqa: BLE001
        primary_ok = False
        lifecycle._atomic_json(
            run_root / "failure.json",
            {
                "stage": stage,
                "error_type": type(exc).__name__,
                "message_recorded": False,
            },
        )
    finally:
        cleanup_ok = lifecycle.cleanup_children(
            lifecycle._ACTIVE_CHILDREN
        )
        lifecycle._ACTIVE_CHILDREN.clear()
        postflight_ok = (
            lifecycle._run_postflight(
                run_root, "postflight_after", args.gpus
            )
            == 0
        )
        if before_preflight and cleanup_ok and postflight_ok:
            after_preflight_ok = (
                lifecycle._run_preflight(
                    run_root, "preflight_after", args.gpus
                )
                == 0
            )
        if after_preflight_ok:
            comparison_ok = (
                lifecycle._run_to_files(
                    [
                        sys.executable,
                        str(ROOT / "tests" / "compare_bi100_preflights.py"),
                        "--preflight",
                        f"before={run_root / 'preflight_before.json'}",
                        "--preflight",
                        f"after={run_root / 'preflight_after.json'}",
                        "--expected-gpus",
                        "1,2,3",
                        "--max-free-memory-drop-bytes",
                        "1073741824",
                        "--out",
                        str(run_root / "preflight_comparison.json"),
                    ],
                    run_root / "preflight_comparison.stdout",
                    run_root / "preflight_comparison.stderr",
                    label="preflight_comparison",
                    timeout_s=300,
                    environment=lifecycle._base_environment(),
                )
                == 0
            )
        fatal = lifecycle._scan_fatal(run_root)
        lifecycle._atomic_json(run_root / "fatal_scan.json", fatal)
        source_unchanged = (
            lifecycle._git("rev-parse", "HEAD") == revision
            and not lifecycle._git(
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                ".",
                ":(exclude)bench_runs/**",
            )
        )
        qualified = all(
            (
                primary_ok,
                screen.get("qualified") is True,
                cleanup_ok,
                postflight_ok,
                after_preflight_ok,
                comparison_ok,
                fatal.get("qualified") is True,
                source_unchanged,
            )
        )
        lifecycle._atomic_json(
            run_root / "runner_status.json",
            {
                "schema": RUNNER_SCHEMA,
                "version": 1,
                "qualified": qualified,
                "terminal_stage": stage,
                "source_revision": revision,
                "instance": args.instance,
                "gpus": args.gpus,
                "fixed_cases": list(CASES),
                "wall_s": time.monotonic() - started,
                "baseline_extension_sha256": baseline_sha,
                "baseline_module_name": args.baseline_module_name,
                "candidate_extension_sha256": candidate_sha,
                "candidate_module_name": args.candidate_module_name,
                "candidate_source": str(candidate_source.relative_to(ROOT)),
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
            },
        )
    return (
        0
        if json.loads(
            (run_root / "runner_status.json").read_text(encoding="ascii")
        )["qualified"]
        else 1
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("baseline_extension", type=Path)
    parser.add_argument("candidate_extension", type=Path)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--baseline-module-name",
        default=BASELINE_MODULE_NAME,
    )
    parser.add_argument(
        "--candidate-module-name",
        default="corex_fused_paged_prefill_fp16_qk",
    )
    parser.add_argument(
        "--candidate-source",
        type=Path,
        default=(
            ROOT
            / "qwen3_6_scripts"
            / "corex_fused_paged_prefill_fp16_qk.cu"
        ),
    )
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
