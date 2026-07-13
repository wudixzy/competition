#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any


COREX_PYTHON = [
    "/usr/local/corex/lib64/python3/dist-packages",
    "/usr/local/corex/lib/python3/dist-packages",
    "/usr/local/lib/python3.10/site-packages",
]
COREX_LIBS = [
    "/usr/local/corex/lib",
    "/usr/local/corex/lib64",
    "/usr/local/corex-3.2.3/lib",
    "/usr/local/corex-3.2.3/lib64",
    "/usr/local/openmpi/lib",
]


def prepend_env(name: str, values: list[str]) -> None:
    existing = os.environ.get(name)
    os.environ[name] = ":".join(values + ([existing] if existing else []))


def setup_corex_env() -> None:
    prepend_env("PYTHONPATH", COREX_PYTHON)
    prepend_env("LD_LIBRARY_PATH", COREX_LIBS)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def worker(rank: int, world_size: int, init_method: str, ipc: bool,
           sizes: list[int], warmup: int, iterations: int, repeats: int,
           queue: mp.Queue) -> None:
    setup_corex_env()
    os.environ["ENABLE_CUSTOM_IPC"] = "1" if ipc else "0"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        import torch
        import torch.distributed as dist
        from ixformer.contrib.torch.extension.ixformer_torch.distributed import (
            create_ixformer_group_from_pg,
        )
        from ixformer.distributed import all_reduce
        from ixformer._C import _distributed as cdist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        group = create_ixformer_group_from_pg()

        parity_tensor = torch.full(
            (2048,), float(rank + 1), device=f"cuda:{rank}",
            dtype=torch.float16,
        )
        all_reduce(parity_tensor, group=group, async_op=True)
        torch.cuda.synchronize()
        expected = float(sum(range(1, world_size + 1)))
        parity_max_abs = float((parity_tensor.float() - expected).abs().max())

        measurements: dict[str, Any] = {}
        for elements in sizes:
            tensor = torch.zeros(
                elements, device=f"cuda:{rank}", dtype=torch.float16,
            )
            for _ in range(warmup):
                all_reduce(tensor, group=group, async_op=True)
            torch.cuda.synchronize()

            trials = []
            for _ in range(repeats):
                dist.barrier()
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(iterations):
                    all_reduce(tensor, group=group, async_op=True)
                torch.cuda.synchronize()
                trials.append(
                    (time.perf_counter() - started) * 1000.0 / iterations,
                )
            measurements[str(elements)] = {
                "median_ms": statistics.median(trials),
                "p10_ms": percentile(trials, 10),
                "p90_ms": percentile(trials, 90),
                "trials_ms": trials,
            }

        ipc_initiated = bool(cdist.ipc.is_initiated())
        shm_bytes = int(cdist.ipc.get_shm_mem_size()) if ipc_initiated else 0
        queue.put({
            "rank": rank,
            "ok": parity_max_abs == 0.0,
            "parity_max_abs": parity_max_abs,
            "ipc_initiated": ipc_initiated,
            "ipc_shm_bytes": shm_bytes,
            "measurements": measurements,
        })
        dist.destroy_process_group()
    except BaseException as exc:  # noqa: BLE001
        queue.put({
            "rank": rank,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    setup_corex_env()
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    init_method = f"tcp://127.0.0.1:{free_port()}"
    processes = [
        ctx.Process(
            target=worker,
            args=(
                rank, args.world_size, init_method, args.ipc, args.sizes,
                args.warmup, args.iterations, args.repeats, queue,
            ),
        )
        for rank in range(args.world_size)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + args.timeout_s
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    timed_out = []
    for rank, process in enumerate(processes):
        if process.is_alive():
            timed_out.append(rank)
            process.terminate()
    for process in processes:
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)

    received = []
    while not queue.empty():
        received.append(queue.get())
    by_rank = {int(item["rank"]): item for item in received}
    results = []
    for rank, process in enumerate(processes):
        item = by_rank.get(rank, {
            "rank": rank,
            "ok": False,
            "error": "timeout" if rank in timed_out else "no result",
        })
        item["exitcode"] = process.exitcode
        results.append(item)

    aggregates = {}
    if all(item.get("ok") for item in results):
        for elements in args.sizes:
            medians = [
                float(item["measurements"][str(elements)]["median_ms"])
                for item in results
            ]
            aggregates[str(elements)] = {
                "rank_median_ms": medians,
                "max_rank_median_ms": max(medians),
                "median_rank_median_ms": statistics.median(medians),
            }

    return {
        "ok": all(item.get("ok") for item in results),
        "mode": "ixformer_ipc" if args.ipc else "ixformer_nccl",
        "config": {
            "world_size": args.world_size,
            "sizes": args.sizes,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "timeout_s": args.timeout_s,
        },
        "timed_out_ranks": timed_out,
        "aggregates": aggregates,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc", action="store_true")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 8192, 65536])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
