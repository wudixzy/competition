#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path
from typing import Any


COREX_PYTHON = [
    "/usr/local/corex-3.2.3/lib64/python3/dist-packages",
    "/usr/local/corex/lib64/python3/dist-packages",
    "/usr/local/corex/lib/python3/dist-packages",
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


def worker(rank: int, world_size: int, init_method: str, elements: int,
           iterations: int, queue: mp.Queue) -> None:
    setup_corex_env()
    os.environ["ENABLE_CUSTOM_IPC"] = "1"
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

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        group = create_ixformer_group_from_pg()
        expected = float(sum(range(1, world_size + 1)))

        warmup = torch.full(
            (elements,), float(rank + 1), device=f"cuda:{rank}",
            dtype=torch.float16)
        for _ in range(3):
            warmup.fill_(float(rank + 1))
            all_reduce(warmup, group=group, async_op=True)
        torch.cuda.synchronize()

        static = torch.full_like(warmup, float(rank + 1))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            all_reduce(static, group=group, async_op=True)
        torch.cuda.synchronize()

        static.fill_(float(rank + 1))
        graph.replay()
        torch.cuda.synchronize()
        graph_max_abs = float((static.float() - expected).abs().max())

        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            static.fill_(float(rank + 1))
            graph.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - started) * 1000.0 / iterations

        queue.put({
            "rank": rank,
            "ok": graph_max_abs == 0.0,
            "graph_max_abs": graph_max_abs,
            "graph_ms": graph_ms,
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
            args=(rank, args.world_size, init_method, args.elements,
                  args.iterations, queue),
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

    return {
        "ok": all(item.get("ok") for item in results),
        "config": {
            "world_size": args.world_size,
            "elements": args.elements,
            "iterations": args.iterations,
            "timeout_s": args.timeout_s,
        },
        "timed_out_ranks": timed_out,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--elements", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
