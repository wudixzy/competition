#!/usr/bin/env python3
"""Run the frozen L1 operator matrix on the available healthy BI100 GPUs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "production_dense_q8176",
    "production_65k_q8176",
    "production_128k_q8176",
    "production_235k_q5616",
)
CELL_SCHEMA = "bi100-m1-55-production-prefill-cell-v1"
RUNNER_SCHEMA = "bi100-m1-142-l1-subset-screen-v1"
TERM_GRACE_S = 60.0
KILL_GRACE_S = 20.0
CELL_TIMEOUT_S = 3600
RELATIVE_L2_LIMIT = 1e-5
MAX_ABS_LIMIT = 1e-3
MIN_PRODUCTION_SPEEDUP = 1.5
SYSTEM_PYTHONPATH = (
    "/usr/local/corex/lib64/python3/dist-packages:"
    "/usr/local/corex/lib/python3/dist-packages"
)
COREX_LD_LIBRARY_PATH = (
    "/usr/local/corex/lib:/usr/local/corex/lib64:"
    "/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:"
    "/usr/local/openmpi/lib"
)
COREX_PATH = (
    "/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:"
    "/usr/local/openmpi/bin:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin"
)
FATAL_PATTERNS = {
    "cuda_error": re.compile(
        r"CUDA error|illegal memory access|device-side assert", re.I),
    "segfault": re.compile(r"SIGSEGV|Fatal Python error", re.I),
    "oom": re.compile(r"out of memory", re.I),
    "corex": re.compile(r"CoreX.*(?:failed|fatal)", re.I),
    "timeout": re.compile(r"Timeout(?:Error|Expired)", re.I),
    "gloo_reset": re.compile(
        r"Gloo.*(?:connectFullMesh failed|connection reset by peer)", re.I),
    "worker_loss": re.compile(
        r"(?:worker|child process).*(?:died|lost|failed|unexpectedly exited)",
        re.I,
    ),
}


class ParentTermination(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


@dataclass
class Child:
    case: str
    gpu: int
    process: subprocess.Popen[Any]
    starttime: int
    started: float


_ACTIVE_CHILDREN: list[Child] = []


def _signal_handler(signum: int, _frame: Any) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise ParentTermination(signum)


def parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(part.strip()) for part in value.split(",")
                if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU indices must be integers") from exc
    if (
        not gpus
        or len(gpus) != len(set(gpus))
        or any(gpu < 0 for gpu in gpus)
    ):
        raise argparse.ArgumentTypeError(
            "GPU indices must be unique non-negative integers")
    if len(gpus) > len(CASES):
        raise argparse.ArgumentTypeError(
            f"at most {len(CASES)} GPUs are useful for this matrix")
    return gpus


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="ascii") as stream:
        json.dump(
            value,
            stream,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_starttime(pid: int) -> int:
    value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    fields = value[value.rfind(")") + 2:].split()
    return int(fields[19])


def _same_process(child: Child) -> bool:
    try:
        return _read_starttime(child.process.pid) == child.starttime
    except (FileNotFoundError, ProcessLookupError):
        return False


def _signal_children(children: list[Child], signum: int) -> bool:
    ok = True
    for child in children:
        if child.process.poll() is not None:
            continue
        if not _same_process(child):
            ok = False
            continue
        try:
            os.killpg(child.process.pid, signum)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                child.process.send_signal(signum)
            except OSError:
                ok = False
    return ok


def _wait_children(children: list[Child], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(child.process.poll() is not None for child in children):
            break
        time.sleep(0.1)
    for child in children:
        if child.process.poll() is not None:
            child.process.wait()
    return all(child.process.poll() is not None for child in children)


def cleanup_children(children: list[Child]) -> bool:
    live = [child for child in children if child.process.poll() is None]
    ok = _signal_children(live, signal.SIGTERM)
    if not _wait_children(live, TERM_GRACE_S):
        survivors = [
            child for child in live if child.process.poll() is None
        ]
        ok = _signal_children(survivors, signal.SIGKILL) and ok
        ok = _wait_children(survivors, KILL_GRACE_S) and ok
    return ok


def _base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": f"{ROOT / 'tests'}:{SYSTEM_PYTHONPATH}",
        "LD_LIBRARY_PATH": COREX_LD_LIBRARY_PATH,
        "PATH": COREX_PATH,
        "PYTHONFAULTHANDLER": "1",
        "PYTHONUNBUFFERED": "1",
    })
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    return environment


def _spawn_managed(
    *,
    label: str,
    gpu: int,
    command: list[str],
    stdout: Any,
    stderr: Any,
    environment: dict[str, str],
) -> Child:
    blocked = {signal.SIGTERM, signal.SIGINT}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            env=environment,
            start_new_session=True,
        )
        child = Child(
            case=label,
            gpu=gpu,
            process=process,
            starttime=_read_starttime(process.pid),
            started=time.monotonic(),
        )
        _ACTIVE_CHILDREN.append(child)
        return child
    except BaseException:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            try:
                process.wait(timeout=TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    process.wait(timeout=KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    pass
        raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _run_to_files(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    label: str,
    timeout_s: int,
    environment: dict[str, str] | None = None,
) -> int:
    wrapped = [
        "timeout", "--foreground", "--signal=TERM", "--kill-after=90s",
        f"{timeout_s}s", *command,
    ]
    with (
        stdout_path.open("wb") as stdout,
        stderr_path.open("wb") as stderr,
    ):
        child = _spawn_managed(
            label=label,
            gpu=-1,
            command=wrapped,
            stdout=stdout,
            stderr=stderr,
            environment=environment or _base_environment(),
        )
        try:
            return child.process.wait()
        finally:
            if child.process.poll() is not None:
                child.process.wait()
                if child in _ACTIVE_CHILDREN:
                    _ACTIVE_CHILDREN.remove(child)


def _run_postflight(run_root: Path, label: str, gpus: list[int]) -> int:
    return _run_to_files(
        [
            sys.executable,
            str(ROOT / "tests" / "service_postflight_gate.py"),
            "--gpus", ",".join(map(str, gpus)),
            "--settle-timeout-s", "90",
            "--clean-samples", "3",
            "--sample-interval-s", "2",
            "--out", str(run_root / f"{label}.json"),
        ],
        run_root / f"{label}.stdout",
        run_root / f"{label}.stderr",
        label=label,
        timeout_s=300,
        environment=_base_environment(),
    )


def _run_preflight(run_root: Path, label: str, gpus: list[int]) -> int:
    return _run_to_files(
        [
            sys.executable,
            str(ROOT / "tests" / "bi100_parallel_preflight.py"),
            "--gpus", ",".join(map(str, gpus)),
            "--timeout-s", "25",
            "--matmul-size", "1024",
            "--json-out", str(run_root / f"{label}.json"),
            "--work-dir", str(run_root / f"{label}-parallel"),
        ],
        run_root / f"{label}.stdout",
        run_root / f"{label}.stderr",
        label=label,
        timeout_s=480,
        environment=_base_environment(),
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
) -> Child:
    stdout_path = run_root / f"{case}.stdout"
    stderr_path = run_root / f"{case}.stderr"
    environment = _base_environment()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    stdout = stdout_path.open("wb")
    stderr = stderr_path.open("wb")
    try:
        return _spawn_managed(
            label=case,
            gpu=gpu,
            command=[
                "timeout", "--foreground", "--signal=TERM",
                "--kill-after=90s", f"{CELL_TIMEOUT_S}s",
                sys.executable,
                str(ROOT / "tests"
                    / "bench_m1_55_production_prefill.py"),
                "--case", case,
                "--extension", str(extension),
                "--expected-extension-sha256", extension_sha,
                "--source-commit", revision,
                "--runtime-identity", "corex-3.2.3-m1-142",
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
    rows: list[dict[str, Any]] = []
    for wave_index in range(
            0, len(assignments), len(gpus)):
        wave_started = time.monotonic()
        wave_assignments = assignments[wave_index:wave_index + len(gpus)]
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
            for case, gpu in wave_assignments
        ]
        while any(child.process.poll() is None for child in children):
            time.sleep(0.1)
        wave_rows = []
        for child in children:
            returncode = child.process.wait()
            elapsed_s = time.monotonic() - child.started
            row = {
                "case": child.case,
                "gpu": child.gpu,
                "returncode": returncode,
                "elapsed_s": elapsed_s,
            }
            rows.append(row)
            wave_rows.append(row)
            _ACTIVE_CHILDREN.remove(child)
        _atomic_json(
            run_root / f"wave-{wave_index // len(gpus)}.json",
            {
                "wave": wave_index // len(gpus),
                "wall_s": time.monotonic() - wave_started,
                "cells": wave_rows,
            },
        )
        if any(row["returncode"] != 0 for row in wave_rows):
            break
    return rows


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_or_none(value: Any) -> int | float | None:
    return value if _finite_number(value) else None


def aggregate_cell_reports(
    *,
    reports: list[dict[str, Any]],
    gpus: list[int],
    extension_sha: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    by_case: dict[str, dict[str, Any]] = {}
    rows = []
    for report in reports:
        case = report.get("case")
        if case not in CASES or case in by_case:
            reasons.append(f"invalid or duplicate case: {case!r}")
            continue
        by_case[case] = report
        numerical = report.get("numerical") or {}
        timings = report.get("timings") or {}
        extension = report.get("extension") or {}
        evaluation = report.get("evaluation") or {}
        gpu = report.get("visible_physical_gpu")
        speedup = timings.get("speedup")
        row = {
            "case": case,
            "gpu": gpu,
            "qualified": evaluation.get("qualified") is True,
            "speedup": _finite_or_none(speedup),
            "output_relative_l2": _finite_or_none(
                numerical.get("output_relative_l2")),
            "lse_relative_l2": _finite_or_none(
                numerical.get("lse_relative_l2")),
            "output_max_abs": _finite_or_none(
                numerical.get("output_max_abs")),
            "finite": numerical.get("finite"),
        }
        rows.append(row)
        if report.get("schema") != CELL_SCHEMA:
            reasons.append(f"{case}: invalid cell schema")
        expected_gpu = gpus[CASES.index(case) % len(gpus)]
        if (
            not isinstance(gpu, int)
            or isinstance(gpu, bool)
            or gpu != expected_gpu
        ):
            reasons.append(
                f"{case}: expected physical GPU {expected_gpu}, got {gpu!r}")
        if extension.get("sha256") != extension_sha:
            reasons.append(f"{case}: extension identity mismatch")
        if evaluation.get("qualified") is not True:
            reasons.append(f"{case}: cell evaluation failed")
        if (
            not _finite_number(speedup)
            or float(speedup) < MIN_PRODUCTION_SPEEDUP
        ):
            reasons.append(
                f"{case}: speedup is below {MIN_PRODUCTION_SPEEDUP:.1f}x")
        if numerical.get("finite") is not True:
            reasons.append(f"{case}: candidate output is nonfinite")
        for field, limit in (
            ("output_relative_l2", RELATIVE_L2_LIMIT),
            ("lse_relative_l2", RELATIVE_L2_LIMIT),
            ("output_max_abs", MAX_ABS_LIMIT),
        ):
            value = numerical.get(field)
            if (
                not _finite_number(value)
                or float(value) < 0
                or float(value) > limit
            ):
                reasons.append(
                    f"{case}: numerical.{field} exceeds {limit:g}")
    missing = [case for case in CASES if case not in by_case]
    if missing:
        reasons.append(f"missing frozen cases: {missing}")
    rows.sort(key=lambda row: CASES.index(row["case"]))
    screen_qualified = not reasons
    full_l1_contract = (
        screen_qualified
        and len(gpus) == len(CASES)
        and len({row["gpu"] for row in rows}) == len(CASES)
    )
    speedups = [
        float(row["speedup"]) for row in rows
        if _finite_number(row["speedup"])
    ]
    return {
        "screen_qualified": screen_qualified,
        "full_l1_contract_satisfied": full_l1_contract,
        "reasons": reasons,
        "rows": rows,
        "minimum_speedup": min(speedups) if speedups else None,
        "authorization": {
            "four_gpu_l1_rerun_authorized": screen_qualified,
            "l2_capture_authorized": full_l1_contract,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }


def _scan_fatal(run_root: Path) -> dict[str, Any]:
    counts = {name: 0 for name in FATAL_PATTERNS}
    for path in sorted(run_root.rglob("*")):
        if path.suffix not in {".stdout", ".stderr"}:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                for name, pattern in FATAL_PATTERNS.items():
                    if pattern.search(line):
                        counts[name] += 1
    return {
        "qualified": not any(counts.values()),
        "category_counts": counts,
        "raw_messages_recorded": False,
    }


def _load_cell_reports(run_root: Path) -> list[dict[str, Any]]:
    reports = []
    for case in CASES:
        path = run_root / f"{case}.json"
        if path.is_file():
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def _validate(args: argparse.Namespace) -> tuple[Path, str, str]:
    run_root = args.run_root.resolve()
    if (
        run_root == Path("/tmp")
        or not run_root.is_relative_to(Path("/tmp"))
        or run_root.exists()
    ):
        raise ValueError("run root must be a new private path under /tmp")
    extension = args.extension.resolve(strict=True)
    if not extension.is_file() or extension.stat().st_size == 0:
        raise ValueError("candidate extension is empty")
    status = _git(
        "status", "--porcelain", "--untracked-files=all", "--", ".",
        ":(exclude)bench_runs/**",
    )
    if status:
        raise RuntimeError("M1-142 requires a clean source tree")
    revision = _git("rev-parse", "HEAD")
    return extension, _sha256(extension), revision


def run(args: argparse.Namespace) -> int:
    global _ACTIVE_CHILDREN
    extension, extension_sha, revision = _validate(args)
    run_root = args.run_root.resolve()
    run_root.mkdir(mode=0o700, parents=True)
    os.chmod(run_root, 0o700)
    started = time.monotonic()
    stage = "initialization"
    before_preflight = False
    cell_rows: list[dict[str, Any]] = []
    primary_ok = True
    cleanup_ok = True
    postflight_ok = False
    after_preflight_ok = False
    comparison_ok = False
    aggregate: dict[str, Any] = {
        "screen_qualified": False,
        "full_l1_contract_satisfied": False,
        "reasons": ["operator screen did not run"],
        "rows": [],
        "authorization": {
            "four_gpu_l1_rerun_authorized": False,
            "l2_capture_authorized": False,
            "main_or_yaml_change_authorized": False,
            "official_score_claim_authorized": False,
        },
    }
    try:
        _atomic_json(run_root / "identity.json", {
            "source_revision": revision,
            "source_branch": _git("branch", "--show-current"),
            "instance": args.instance,
            "gpus": args.gpus,
            "extension_sha256": extension_sha,
            "kernel_source_sha256": _sha256(
                ROOT / "qwen3_6_scripts"
                / "corex_fused_paged_prefill_split4.cu"),
        })
        stage = "postflight_before"
        if _run_postflight(run_root, stage, args.gpus) != 0:
            raise RuntimeError("initial GPU postflight failed")
        stage = "preflight_before"
        if _run_preflight(run_root, stage, args.gpus) != 0:
            raise RuntimeError("initial GPU preflight failed")
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
            raise RuntimeError("one or more operator cells failed")
        aggregate = aggregate_cell_reports(
            reports=_load_cell_reports(run_root),
            gpus=args.gpus,
            extension_sha=extension_sha,
        )
        _atomic_json(run_root / "screen.json", aggregate)
        if not aggregate["screen_qualified"]:
            raise RuntimeError("operator screen did not qualify")
    except ParentTermination:
        primary_ok = False
        raise
    except BaseException as exc:  # noqa: BLE001 - preserve terminal stage.
        primary_ok = False
        _atomic_json(run_root / "failure.json", {
            "stage": stage,
            "error_type": type(exc).__name__,
            "message_recorded": False,
        })
    finally:
        cleanup_ok = cleanup_children(_ACTIVE_CHILDREN)
        _ACTIVE_CHILDREN = []
        postflight_ok = (
            _run_postflight(run_root, "postflight_after", args.gpus) == 0)
        if before_preflight and cleanup_ok and postflight_ok:
            after_preflight_ok = (
                _run_preflight(
                    run_root, "preflight_after", args.gpus) == 0)
        if after_preflight_ok:
            comparison_ok = _run_to_files(
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
                environment=_base_environment(),
            ) == 0
        fatal = _scan_fatal(run_root)
        _atomic_json(run_root / "fatal_scan.json", fatal)
        source_unchanged = (
            _git("rev-parse", "HEAD") == revision
            and not _git(
                "status", "--porcelain", "--untracked-files=all", "--",
                ".", ":(exclude)bench_runs/**")
        )
        qualified = all((
            primary_ok,
            aggregate.get("screen_qualified") is True,
            cleanup_ok,
            fatal.get("qualified") is True,
            postflight_ok,
            after_preflight_ok,
            comparison_ok,
            source_unchanged,
        ))
        _atomic_json(run_root / "runner_status.json", {
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
            "screen": aggregate,
            "lifecycle": {
                "cleanup_reaped": cleanup_ok,
                "fatal_scan_qualified": fatal.get("qualified"),
                "postflight_qualified": postflight_ok,
                "after_preflight_qualified": after_preflight_ok,
                "preflight_comparison_qualified": comparison_ok,
                "source_unchanged": source_unchanged,
            },
            "authorization": aggregate.get("authorization"),
            "privacy": {
                "prompts_recorded": False,
                "model_outputs_recorded": False,
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
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    args = parser.parse_args()
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in previous_handlers:
        signal.signal(signum, _signal_handler)
    try:
        return run(args)
    except ParentTermination as termination:
        cleanup_children(_ACTIVE_CHILDREN)
        return 128 + termination.signum
    finally:
        cleanup_children(_ACTIVE_CHILDREN)
        _ACTIVE_CHILDREN.clear()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
