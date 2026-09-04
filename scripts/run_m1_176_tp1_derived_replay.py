#!/usr/bin/env python3
"""Derive and replay four TP4 rank-local banks from one private TP1 capture."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
from typing import Any

import run_m1_142_l1_subset_screen as lifecycle


ROOT = Path(__file__).resolve().parents[1]
DERIVE = ROOT / "scripts" / "derive_m1_176_tp1_rank0_activation_bank.py"
REPLAY = ROOT / "tests" / "replay_m1_176_tp1_rank0_activation.py"
RANKS = (0, 1, 2, 3)
CELL_TIMEOUT_S = 7200
RUNTIME_SCHEMA = "bi100-m1-176-four-rank-real-activation-replay-v2"


def _private_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.resolve().is_relative_to(Path("/tmp"))
        and not path.stat().st_mode & 0o077
        and not path.parent.stat().st_mode & 0o077
    )


def _run_checked(
    command: list[str],
    run_root: Path,
    label: str,
    timeout_s: int,
) -> None:
    rc = lifecycle._run_to_files(
        command,
        run_root / f"{label}.stdout",
        run_root / f"{label}.stderr",
        label=label,
        timeout_s=timeout_s,
        environment=lifecycle._base_environment(),
    )
    lifecycle._atomic_json(run_root / f"{label}.rc.json", {"returncode": rc})
    if rc != 0:
        raise RuntimeError(f"{label} failed")


def _run_nccl(run_root: Path, label: str, gpus: list[int]) -> None:
    _run_checked(
        [
            sys.executable,
            str(ROOT / "tests" / "bi100_nccl_preflight.py"),
            "--gpus", ",".join(map(str, gpus)),
            "--timeout-s", "60",
            "--json-out", str(run_root / f"{label}.json"),
        ],
        run_root,
        label,
        300,
    )


def _derive_banks(
    source_manifest: Path,
    run_root: Path,
    revision: str,
    runtime_identity: str,
) -> list[Path]:
    manifests = []
    for rank in RANKS:
        output_dir = run_root / f"rank-{rank}-bank"
        report = run_root / f"rank-{rank}-derive.json"
        _run_checked(
            [
                sys.executable,
                str(DERIVE),
                "--source-manifest", str(source_manifest),
                "--output-dir", str(output_dir),
                "--expected-source-revision", revision,
                "--expected-runtime-identity", runtime_identity,
                "--logical-rank", str(rank),
                "--report", str(report),
            ],
            run_root,
            f"rank-{rank}-derive",
            1800,
        )
        value = json.loads(report.read_text(encoding="ascii"))
        manifest = Path(value["manifest"]).resolve(strict=True)
        if not _private_file(manifest):
            raise RuntimeError("derived manifest is not private")
        manifests.append(manifest)
    return manifests


def _spawn_replay(
    rank: int,
    gpu: int,
    manifest: Path,
    baseline: Path,
    candidate: Path,
    baseline_sha: str,
    candidate_sha: str,
    revision: str,
    runtime_identity: str,
    instance: str,
    run_root: Path,
) -> lifecycle.Child:
    environment = lifecycle._base_environment()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    stdout = (run_root / f"rank-{rank}-replay.stdout").open("wb")
    stderr = (run_root / f"rank-{rank}-replay.stderr").open("wb")
    try:
        return lifecycle._spawn_managed(
            label=f"rank-{rank}-replay",
            gpu=gpu,
            command=[
                "timeout", "--foreground", "--signal=TERM",
                "--kill-after=90s", f"{CELL_TIMEOUT_S}s",
                sys.executable,
                str(REPLAY),
                "--bank-manifest", str(manifest),
                "--baseline-extension", str(baseline),
                "--expected-baseline-sha256", baseline_sha,
                "--baseline-module-name", "corex_fused_paged_prefill",
                "--candidate-extension", str(candidate),
                "--expected-candidate-sha256", candidate_sha,
                "--candidate-module-name",
                "corex_fused_paged_prefill_fp16_qk",
                "--capture-source-revision", revision,
                "--baseline-source-revision", revision,
                "--candidate-source-revision", revision,
                "--runtime-identity", runtime_identity,
                "--instance", instance,
                "--visible-physical-gpu", str(gpu),
                "--logical-tp-rank", str(rank),
                "--out", str(run_root / f"rank-{rank}-replay.json"),
            ],
            stdout=stdout,
            stderr=stderr,
            environment=environment,
        )
    finally:
        stdout.close()
        stderr.close()


def _run_replays(
    manifests: list[Path],
    gpus: list[int],
    baseline: Path,
    candidate: Path,
    baseline_sha: str,
    candidate_sha: str,
    revision: str,
    runtime_identity: str,
    instance: str,
    run_root: Path,
) -> list[dict[str, Any]]:
    children = [
        _spawn_replay(
            rank, gpu, manifest, baseline, candidate,
            baseline_sha, candidate_sha, revision, runtime_identity,
            instance, run_root)
        for rank, gpu, manifest in zip(RANKS, gpus, manifests)
    ]
    while any(child.process.poll() is None for child in children):
        time.sleep(1.0)
    rows = []
    for child in children:
        rows.append({
            "logical_tp_rank": int(child.case.split("-")[1]),
            "visible_physical_gpu": child.gpu,
            "returncode": child.process.wait(),
            "elapsed_s": time.monotonic() - child.started,
        })
        lifecycle._ACTIVE_CHILDREN.remove(child)
    return rows


def _aggregate(
    run_root: Path,
    gpus: list[int],
    revision: str,
    runtime_identity: str,
    baseline_sha: str,
    candidate_sha: str,
) -> dict[str, Any]:
    invalid_reasons = []
    g2_reasons = []
    rows = []
    for rank, gpu in zip(RANKS, gpus):
        path = run_root / f"rank-{rank}-replay.json"
        if not path.is_file():
            invalid_reasons.append(f"rank {rank}: replay report missing")
            continue
        report = json.loads(path.read_text(encoding="ascii"))
        records = report.get("records")
        contract_valid = (
            report.get("schema")
            == "bi100-m1-176-tp1-derived-rank-replay-v2"
            and report.get("version") == 2
            and report.get("logical_tp_rank") == rank
            and report.get("visible_physical_gpu") == gpu
            and report.get("capture_source_revision") == revision
            and report.get("baseline_source_revision") == revision
            and report.get("candidate_source_revision") == revision
            and report.get("runtime_identity") == runtime_identity
            and (report.get("baseline_extension") or {}).get("sha256")
            == baseline_sha
            and (report.get("candidate_extension") or {}).get("sha256")
            == candidate_sha
            and isinstance(records, list)
            and len(records) == 3
        )
        if not contract_valid:
            invalid_reasons.append(f"rank {rank}: replay contract differs")
            continue
        if report.get("all_qualified") is not True:
            g2_reasons.append(f"rank {rank}: calibrated G2 gate failed")
        speedups = [
            float(record["timing"]["order_balanced_geometric_speedup"])
            for record in records
        ]
        rows.append({
            "logical_tp_rank": rank,
            "visible_physical_gpu": gpu,
            "record_count": len(records),
            "all_g2_qualified": report.get("all_qualified") is True,
            "median_kernel_speedup": statistics.median(speedups),
            "minimum_kernel_speedup": min(speedups),
            "records": [{
                "context_tokens": record["context_tokens"],
                "query_length": record["query_length"],
                "relative_l2_error_ratio": record[
                    "candidate_numeric"]["relative_l2_error_ratio"],
                "maximum_absolute_error_ratio": record[
                    "candidate_numeric"]["maximum_absolute_error_ratio"],
                "candidate_lse_relative_l2": record[
                    "candidate_numeric"]["candidate_lse_relative_l2"],
                "kernel_speedup": record[
                    "timing"]["order_balanced_geometric_speedup"],
            } for record in records],
        })
    qualified = not invalid_reasons and not g2_reasons and len(rows) == 4
    result_status = (
        "pass" if qualified else "invalid" if invalid_reasons else "fail")
    return {
        "qualified": qualified,
        "result_status": result_status,
        "invalid_reasons": invalid_reasons,
        "g2_reasons": g2_reasons,
        "four_rank_replay_complete": not invalid_reasons and len(rows) == 4,
        "tp4_model_execution_claimed": False,
        "rows": rows,
    }


def _validate(args: argparse.Namespace) -> tuple[Path, Path, Path, str, str, str]:
    if args.gpus != [0, 1, 2, 3]:
        raise ValueError("M1-176 requires the fixed four-rank GPU mapping 0,1,2,3")
    run_root = args.run_root.resolve()
    if (
        run_root == Path("/tmp")
        or not run_root.is_relative_to(Path("/tmp"))
        or run_root.exists()
    ):
        raise ValueError("run root must be a new private path under /tmp")
    source_manifest = args.source_manifest.resolve(strict=True)
    baseline = args.baseline_extension.resolve(strict=True)
    candidate = args.candidate_extension.resolve(strict=True)
    if not all(_private_file(path) for path in (source_manifest, baseline, candidate)):
        raise ValueError("capture and extensions must be private files under /tmp")
    if baseline == candidate:
        raise ValueError("baseline and candidate artifacts must differ")
    if lifecycle._git(
        "status", "--porcelain", "--untracked-files=all", "--", ".",
        ":(exclude)bench_runs/**",
    ):
        raise RuntimeError("M1-176 requires a clean source tree")
    revision = lifecycle._git("rev-parse", "HEAD")
    if not args.runtime_identity:
        raise ValueError("runtime identity is required")
    return (
        source_manifest, baseline, candidate, revision,
        lifecycle._sha256(baseline), lifecycle._sha256(candidate))


def run(args: argparse.Namespace) -> int:
    global_started = time.monotonic()
    source, baseline, candidate, revision, baseline_sha, candidate_sha = (
        _validate(args))
    run_root = args.run_root.resolve()
    run_root.mkdir(mode=0o700, parents=True)
    os.chmod(run_root, 0o700)
    stage = "initialization"
    before_preflight = False
    primary_ok = True
    cleanup_ok = True
    postflight_ok = False
    after_preflight_ok = False
    comparison_ok = False
    replay_processes: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {
        "qualified": False,
        "result_status": "invalid",
        "reasons": ["replay did not run"],
        "rows": [],
    }
    lifecycle._atomic_json(run_root / "identity.json", {
        "source_revision": revision,
        "source_branch": lifecycle._git("branch", "--show-current"),
        "runtime_identity": args.runtime_identity,
        "instance": args.instance,
        "gpus": args.gpus,
        "source_manifest_sha256": lifecycle._sha256(source),
        "baseline_extension_sha256": baseline_sha,
        "candidate_extension_sha256": candidate_sha,
        "fixed_logical_rank_mapping": dict(zip(map(str, RANKS), args.gpus)),
    })
    try:
        stage = "postflight_before"
        if lifecycle._run_postflight(run_root, stage, args.gpus) != 0:
            raise RuntimeError("initial service postflight failed")
        stage = "preflight_before"
        if lifecycle._run_preflight(run_root, stage, args.gpus) != 0:
            raise RuntimeError("initial GPU preflight failed")
        _run_nccl(run_root, "nccl_before", args.gpus)
        before_preflight = True
        stage = "derive_four_rank_banks"
        manifests = _derive_banks(
            source, run_root, revision, args.runtime_identity)
        stage = "parallel_four_rank_replay"
        replay_processes = _run_replays(
            manifests, args.gpus, baseline, candidate,
            baseline_sha, candidate_sha, revision, args.runtime_identity,
            args.instance, run_root)
        aggregate = _aggregate(
            run_root, args.gpus, revision, args.runtime_identity,
            baseline_sha, candidate_sha)
        lifecycle._atomic_json(run_root / "aggregate.json", aggregate)
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
        cleanup_ok = lifecycle.cleanup_children(lifecycle._ACTIVE_CHILDREN)
        lifecycle._ACTIVE_CHILDREN.clear()
        postflight_ok = (
            lifecycle._run_postflight(run_root, "postflight_after", args.gpus)
            == 0)
        if before_preflight and cleanup_ok and postflight_ok:
            after_preflight_ok = (
                lifecycle._run_preflight(
                    run_root, "preflight_after", args.gpus) == 0)
        if after_preflight_ok:
            try:
                _run_nccl(run_root, "nccl_after", args.gpus)
                _run_checked([
                    sys.executable,
                    str(ROOT / "tests" / "compare_bi100_preflights.py"),
                    "--preflight", f"before={run_root / 'preflight_before.json'}",
                    "--preflight", f"after={run_root / 'preflight_after.json'}",
                    "--expected-gpus", "0,1,2,3",
                    "--max-free-memory-drop-bytes", "1073741824",
                    "--out", str(run_root / "preflight_comparison.json"),
                ], run_root, "preflight_comparison", 300)
                comparison_ok = True
            except RuntimeError:
                comparison_ok = False
        fatal = lifecycle._scan_fatal(run_root)
        lifecycle._atomic_json(run_root / "fatal_scan.json", fatal)
        source_unchanged = (
            lifecycle._git("rev-parse", "HEAD") == revision
            and not lifecycle._git(
                "status", "--porcelain", "--untracked-files=all", "--", ".",
                ":(exclude)bench_runs/**"))
        lifecycle_qualified = all((
            cleanup_ok, postflight_ok, after_preflight_ok, comparison_ok,
            fatal.get("qualified") is True, source_unchanged))
        qualified = bool(
            primary_ok and aggregate.get("qualified") is True
            and lifecycle_qualified)
        status = (
            "pass" if qualified
            else "invalid"
            if (not lifecycle_qualified or not primary_ok
                or aggregate.get("result_status") == "invalid")
            else aggregate.get("result_status", "invalid"))
        lifecycle._atomic_json(run_root / "runner_status.json", {
            "schema": RUNTIME_SCHEMA,
            "version": 2,
            "qualified": qualified,
            "result_status": status,
            "terminal_stage": stage,
            "source_revision": revision,
            "runtime_identity": args.runtime_identity,
            "instance": args.instance,
            "gpus": args.gpus,
            "wall_s": time.monotonic() - global_started,
            "replay_processes": replay_processes,
            "aggregate": aggregate,
            "lifecycle": {
                "cleanup_reaped": cleanup_ok,
                "postflight_qualified": postflight_ok,
                "after_preflight_qualified": after_preflight_ok,
                "preflight_comparison_qualified": comparison_ok,
                "fatal_scan_qualified": fatal.get("qualified"),
                "source_unchanged": source_unchanged,
            },
            "privacy": {
                "raw_tensors_confined_to_private_tmp": True,
                "raw_tensors_in_status": False,
                "prompts_in_status": False,
                "model_outputs_in_status": False,
                "credentials_in_status": False,
            },
            "authorization": {
                "l3_short_tp4_authorized": qualified,
                "long_context_or_formal_score_authorized": False,
                "main_or_yaml_change_authorized": False,
            },
        })
    result = json.loads(
        (run_root / "runner_status.json").read_text(encoding="ascii"))
    return 0 if result["qualified"] else 1


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("baseline_extension", type=Path)
    parser.add_argument("candidate_extension", type=Path)
    parser.add_argument("runtime_identity")
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--gpus", type=lifecycle.parse_gpus,
        default=lifecycle.parse_gpus("0,1,2,3"))
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
