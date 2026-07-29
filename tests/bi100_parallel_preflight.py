#!/usr/bin/env python3
"""Run independent BI100 single-GPU preflights concurrently."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


SCHEMA = "bi100-gpu-preflight-v1"
TERM_GRACE_S = 60.0
KILL_GRACE_S = 20.0
_ACTIVE_PROCESSES: list[subprocess.Popen[Any]] = []


class ParentTermination(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _signal_handler(signum: int, _frame: Any) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise ParentTermination(signum)


def parse_gpus(value: str) -> list[int]:
    gpus = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not gpus:
        raise argparse.ArgumentTypeError("at least one GPU index is required")
    if len(gpus) != len(set(gpus)):
        raise argparse.ArgumentTypeError("GPU indices must be unique")
    return gpus


def _signal_groups(
    processes: list[subprocess.Popen[Any]],
    signum: int,
) -> None:
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        except OSError:
            process.send_signal(signum)


def _wait_for_groups(
    processes: list[subprocess.Popen[Any]],
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes):
            break
        time.sleep(0.1)
    for process in processes:
        if process.poll() is not None:
            process.wait()
    return all(process.poll() is not None for process in processes)


def cleanup_process_groups(
    processes: list[subprocess.Popen[Any]],
) -> bool:
    live = [process for process in processes if process.poll() is None]
    _signal_groups(live, signal.SIGTERM)
    if not _wait_for_groups(live, TERM_GRACE_S):
        survivors = [
            process for process in live if process.poll() is None
        ]
        _signal_groups(survivors, signal.SIGKILL)
        if not _wait_for_groups(survivors, KILL_GRACE_S):
            return False
    return True


def _read_child_result(
    result_path: Path,
    gpu: int,
    returncode: int | None,
) -> dict[str, Any]:
    if result_path.is_file():
        try:
            value = json.loads(result_path.read_text(encoding="utf-8"))
            results = value.get("results")
            if (
                isinstance(results, list)
                and len(results) == 1
                and isinstance(results[0], dict)
            ):
                result = dict(results[0])
                result["subrunner_returncode"] = returncode
                return result
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    return {
        "gpu": gpu,
        "ok": False,
        "stage": "subrunner_result_missing",
        "subrunner_returncode": returncode,
    }


def run_parallel(
    *,
    gpus: list[int],
    timeout_s: float,
    matmul_size: int,
    serial_script: Path,
    work_dir: Path,
) -> dict[str, Any]:
    global _ACTIVE_PROCESSES
    started = time.monotonic()
    if work_dir.exists():
        if any(work_dir.iterdir()):
            raise FileExistsError(
                f"parallel preflight work directory is not empty: "
                f"{work_dir}")
    else:
        work_dir.mkdir(mode=0o700, parents=True)
    os.chmod(work_dir, 0o700)
    records: list[dict[str, Any]] = []
    blocked = {signal.SIGTERM, signal.SIGINT}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        for gpu in gpus:
            result_path = work_dir / f"gpu{gpu}.json"
            stdout_path = work_dir / f"gpu{gpu}.stdout"
            stderr_path = work_dir / f"gpu{gpu}.stderr"
            stdout_stream = stdout_path.open("wb")
            stderr_stream = stderr_path.open("wb")
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(serial_script),
                        "--gpus", str(gpu),
                        "--timeout-s", str(timeout_s),
                        "--matmul-size", str(matmul_size),
                        "--json-out", str(result_path),
                    ],
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    start_new_session=True,
                )
            finally:
                stdout_stream.close()
                stderr_stream.close()
            _ACTIVE_PROCESSES.append(process)
            records.append({
                "gpu": gpu,
                "process": process,
                "result_path": result_path,
            })
            print(json.dumps({
                "event": "probe_started",
                "gpu": gpu,
                "pid": process.pid,
            }, sort_keys=True), flush=True)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    child_budget_s = timeout_s + TERM_GRACE_S + KILL_GRACE_S + 15.0
    deadline = time.monotonic() + child_budget_s
    completed: set[int] = set()
    timed_out = False
    while len(completed) != len(records):
        for record in records:
            gpu = int(record["gpu"])
            process = record["process"]
            if gpu not in completed and process.poll() is not None:
                completed.add(gpu)
                print(json.dumps({
                    "event": "probe_finished",
                    "gpu": gpu,
                    "returncode": process.returncode,
                }, sort_keys=True), flush=True)
        if len(completed) == len(records):
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(0.1)

    cleanup_reaped = cleanup_process_groups(_ACTIVE_PROCESSES)
    results = [
        _read_child_result(
            record["result_path"],
            int(record["gpu"]),
            record["process"].returncode,
        )
        for record in records
    ]
    _ACTIVE_PROCESSES = []
    return {
        "schema": SCHEMA,
        "version": 1,
        "runner": "parallel-single-gpu",
        "ok": (
            not timed_out
            and cleanup_reaped
            and all(result.get("ok") for result in results)
        ),
        "parallel": True,
        "gpus": gpus,
        "timeout_s": timeout_s,
        "matmul_size": matmul_size,
        "wall_s": time.monotonic() - started,
        "runner_timed_out": timed_out,
        "cleanup_reaped": cleanup_reaped,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe independent BI100 GPUs in parallel.")
    parser.add_argument(
        "--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    parser.add_argument("--timeout-s", type=float, default=25.0)
    parser.add_argument("--matmul-size", type=int, default=1024)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    serial_script = Path(__file__).with_name("bi100_preflight.py")

    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in previous_handlers:
        signal.signal(signum, _signal_handler)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="bi100-parallel-preflight-")
        work_dir = Path(temporary.name)
    else:
        work_dir = args.work_dir
    try:
        summary = run_parallel(
            gpus=args.gpus,
            timeout_s=args.timeout_s,
            matmul_size=args.matmul_size,
            serial_script=serial_script,
            work_dir=work_dir,
        )
    except ParentTermination as termination:
        cleanup_ok = cleanup_process_groups(_ACTIVE_PROCESSES)
        print(
            f"parallel GPU preflight interrupted by signal "
            f"{termination.signum}; "
            f"child_cleanup_ok={str(cleanup_ok).lower()}",
            file=sys.stderr,
            flush=True,
        )
        return 128 + termination.signum
    finally:
        cleanup_process_groups(_ACTIVE_PROCESSES)
        _ACTIVE_PROCESSES.clear()
        if temporary is not None:
            temporary.cleanup()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    for result in summary["results"]:
        status = "PASS" if result.get("ok") else "FAIL"
        print(
            f"[{status}] gpu={result.get('gpu')} "
            f"{json.dumps(result, sort_keys=True, ensure_ascii=False)}",
            flush=True,
        )
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False), flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True,
                       ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
