#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_module
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any


COREX_BIN_PATHS = [
    "/usr/local/corex/bin",
    "/usr/local/corex-3.2.3/bin",
]
COREX_LIBRARY_PATHS = [
    "/usr/local/corex/lib",
    "/usr/local/corex/lib64",
    "/usr/local/corex-3.2.3/lib",
    "/usr/local/corex-3.2.3/lib64",
    "/usr/local/openmpi/lib",
]
COREX_PYTHON_PATHS = [
    "/usr/local/corex/lib64/python3/dist-packages",
    "/usr/local/corex/lib/python3/dist-packages",
]
TERM_GRACE_S = 60.0
KILL_GRACE_S = 20.0
_ACTIVE_PROCESSES: list[mp.Process] = []


class ParentTermination(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _parent_signal_handler(signum: int, _frame: Any) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise ParentTermination(signum)


def _prepend_env_list(key: str, values: list[str]) -> None:
    existing = os.environ.get(key, "")
    parts = values + ([existing] if existing else [])
    os.environ[key] = ":".join(parts)


def setup_corex_env() -> None:
    _prepend_env_list("PATH", COREX_BIN_PATHS)
    _prepend_env_list("LD_LIBRARY_PATH", COREX_LIBRARY_PATHS)
    _prepend_env_list("PYTHONPATH", COREX_PYTHON_PATHS)
    # multiprocessing.spawn copies the parent's live sys.path into each child;
    # changing PYTHONPATH alone after interpreter startup is insufficient.
    for path in reversed(COREX_PYTHON_PATHS):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def parse_gpus(value: str) -> list[int]:
    gpus = []
    for part in value.split(","):
        part = part.strip()
        if part:
            gpus.append(int(part))
    if len(gpus) < 2:
        raise argparse.ArgumentTypeError("at least two GPU indices are required")
    return gpus


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def worker(rank: int,
           gpu: int,
           world_size: int,
           init_method: str,
           queue: mp.Queue) -> None:
    signal.pthread_sigmask(
        signal.SIG_UNBLOCK,
        {signal.SIGTERM, signal.SIGINT},
    )
    os.setsid()
    setup_corex_env()
    try:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(gpu)
        tensor = torch.tensor([float(rank + 1)], device=f"cuda:{gpu}")
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        value = float(tensor.item())
        dist.destroy_process_group()
        queue.put({
            "rank": rank,
            "gpu": gpu,
            "ok": True,
            "value": value,
        })
    except BaseException as exc:  # noqa: BLE001 - preserve diagnostic text.
        queue.put({
            "rank": rank,
            "gpu": gpu,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def _signal_process_group(process: mp.Process, signum: int) -> None:
    if process.pid is None or not process.is_alive():
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return
    except OSError:
        if signum == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def _join_processes(processes: list[mp.Process], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))


def _cleanup_processes(processes: list[mp.Process]) -> dict[str, Any]:
    term_ranks = [
        rank for rank, process in enumerate(processes)
        if process.is_alive()
    ]
    for rank in term_ranks:
        _signal_process_group(processes[rank], signal.SIGTERM)
    _join_processes(
        [processes[rank] for rank in term_ranks],
        TERM_GRACE_S,
    )

    kill_ranks = [
        rank for rank in term_ranks
        if processes[rank].is_alive()
    ]
    for rank in kill_ranks:
        _signal_process_group(processes[rank], signal.SIGKILL)
    _join_processes(
        [processes[rank] for rank in kill_ranks],
        KILL_GRACE_S,
    )

    surviving_ranks = [
        rank for rank, process in enumerate(processes)
        if process.is_alive()
    ]
    return {
        "term_grace_s": TERM_GRACE_S,
        "kill_grace_s": KILL_GRACE_S,
        "term_ranks": term_ranks,
        "kill_ranks": kill_ranks,
        "surviving_ranks": surviving_ranks,
        "reaped": not surviving_ranks,
    }


def _cleanup_active_processes() -> dict[str, Any]:
    global _ACTIVE_PROCESSES
    report = _cleanup_processes(_ACTIVE_PROCESSES)
    if report["reaped"]:
        _ACTIVE_PROCESSES = []
    return report


def _start_process(process: mp.Process) -> None:
    global _ACTIVE_PROCESSES
    blocked = {signal.SIGTERM, signal.SIGINT}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        process.start()
        _ACTIVE_PROCESSES.append(process)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def run_probe(gpus: list[int], timeout_s: float) -> dict[str, Any]:
    global _ACTIVE_PROCESSES
    if _ACTIVE_PROCESSES:
        raise RuntimeError("previous NCCL probe processes are still active")
    setup_corex_env()
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    init_method = f"tcp://127.0.0.1:{free_port()}"
    world_size = len(gpus)
    processes = [
        ctx.Process(
            target=worker,
            args=(rank, gpu, world_size, init_method, queue),
        )
        for rank, gpu in enumerate(gpus)
    ]

    try:
        for process in processes:
            _start_process(process)

        _join_processes(processes, timeout_s)
        timed_out = [
            rank for rank, process in enumerate(processes)
            if process.is_alive()
        ]
        cleanup = _cleanup_processes(processes)
        if cleanup["reaped"]:
            _ACTIVE_PROCESSES = []
    except BaseException:
        _cleanup_active_processes()
        raise

    rank_results: list[dict[str, Any]] = []
    while True:
        try:
            rank_results.append(queue.get_nowait())
        except queue_module.Empty:
            break
    queue.close()
    queue.join_thread()
    rank_results_by_rank = {int(item["rank"]): item for item in rank_results}
    results = []
    expected_sum = float(sum(range(1, world_size + 1)))
    for rank, gpu in enumerate(gpus):
        item = rank_results_by_rank.get(rank)
        if item is None:
            item = {
                "rank": rank,
                "gpu": gpu,
                "ok": False,
                "error": "timeout" if rank in timed_out else "no result",
            }
        elif item.get("ok") and item.get("value") != expected_sum:
            item["ok"] = False
            item["error"] = f"unexpected all_reduce value {item.get('value')}"
        item["exitcode"] = processes[rank].exitcode
        results.append(item)

    return {
        "ok": all(item.get("ok") for item in results),
        "expected_sum": expected_sum,
        "init_method": init_method,
        "results": results,
        "timed_out_ranks": timed_out,
        "cleanup": cleanup,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a timed BI100 NCCL all_reduce preflight.")
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in previous_handlers:
        signal.signal(signum, _parent_signal_handler)
    try:
        summary = run_probe(args.gpus, args.timeout_s)
    except ParentTermination as termination:
        cleanup = _cleanup_active_processes()
        print(
            f"NCCL preflight interrupted by signal {termination.signum}; "
            f"child_cleanup_reaped={str(cleanup['reaped']).lower()}",
            file=sys.stderr,
            flush=True,
        )
        return 128 + termination.signum
    finally:
        _cleanup_active_processes()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    for result in summary["results"]:
        status = "PASS" if result.get("ok") else "FAIL"
        print(
            f"[{status}] rank={result.get('rank')} gpu={result.get('gpu')} "
            f"{json.dumps(result, sort_keys=True, ensure_ascii=False)}",
            flush=True,
        )
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False), flush=True)
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
