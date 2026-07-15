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


def _prepend_env_list(key: str, values: list[str]) -> None:
    existing = os.environ.get(key, "")
    os.environ[key] = ":".join(values + ([existing] if existing else []))


def setup_corex_env() -> None:
    _prepend_env_list("PATH", COREX_BIN_PATHS)
    _prepend_env_list("LD_LIBRARY_PATH", COREX_LIBRARY_PATHS)
    _prepend_env_list("PYTHONPATH", COREX_PYTHON_PATHS)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def worker(rank: int, world_size: int, init_method: str,
           queue: mp.Queue) -> None:
    setup_corex_env()

    def stage(name: str) -> None:
        queue.put({"kind": "stage", "rank": rank, "stage": name})

    try:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        tensor = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
        stage("world_nccl_start")
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        stage("world_nccl_done")

        ranks = list(range(world_size))
        groups: list[Any] = []
        for label in ("world_coordinator", "tensor_parallel"):
            stage(f"{label}_nccl_start")
            device_group = dist.new_group(ranks, backend="nccl")
            stage(f"{label}_nccl_done")
            stage(f"{label}_gloo_start")
            cpu_group = dist.new_group(ranks, backend="gloo")
            stage(f"{label}_gloo_done")
            dist.barrier(group=cpu_group)
            stage(f"{label}_gloo_barrier_done")
            groups.extend((device_group, cpu_group))

        own_pipeline_groups = []
        for pipeline_rank in ranks:
            pipeline_ranks = [pipeline_rank]
            stage(f"pipeline_{pipeline_rank}_nccl_start")
            device_group = dist.new_group(pipeline_ranks, backend="nccl")
            stage(f"pipeline_{pipeline_rank}_nccl_done")
            stage(f"pipeline_{pipeline_rank}_gloo_start")
            cpu_group = dist.new_group(pipeline_ranks, backend="gloo")
            stage(f"pipeline_{pipeline_rank}_gloo_done")
            groups.extend((device_group, cpu_group))
            if rank == pipeline_rank:
                own_pipeline_groups.append((device_group, cpu_group))

        dist.all_reduce(tensor, group=groups[2])
        torch.cuda.synchronize()
        value = float(tensor.item())
        for _, cpu_group in own_pipeline_groups:
            dist.barrier(group=cpu_group)
        stage("collectives_done")
        dist.destroy_process_group()
        queue.put({
            "kind": "result",
            "rank": rank,
            "gpu": rank,
            "ok": True,
            "value": value,
            "stage": "done",
        })
    except BaseException as exc:  # noqa: BLE001 - preserve diagnostics.
        queue.put({
            "kind": "result",
            "rank": rank,
            "gpu": rank,
            "ok": False,
            "stage": "exception",
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def run_probe(world_size: int, timeout_s: float) -> dict[str, Any]:
    setup_corex_env()
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    init_method = f"tcp://127.0.0.1:{free_port()}"
    processes = [
        ctx.Process(target=worker, args=(rank, world_size, init_method, queue))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()

    deadline = time.monotonic() + timeout_s
    events: list[dict[str, Any]] = []
    while time.monotonic() < deadline and any(
            process.is_alive() for process in processes):
        while not queue.empty():
            events.append(queue.get())
        time.sleep(0.05)

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
    while not queue.empty():
        events.append(queue.get())

    last_stage = {rank: "not_started" for rank in range(world_size)}
    results_by_rank: dict[int, dict[str, Any]] = {}
    for event in events:
        rank = int(event["rank"])
        if event["kind"] == "stage":
            last_stage[rank] = str(event["stage"])
        else:
            results_by_rank[rank] = event

    expected_sum = float(sum(range(1, world_size + 1)))
    results = []
    for rank, process in enumerate(processes):
        result = results_by_rank.get(rank, {
            "kind": "result",
            "rank": rank,
            "gpu": rank,
            "ok": False,
            "stage": "timeout" if rank in timed_out else "no_result",
        })
        result["last_stage"] = last_stage[rank]
        result["exitcode"] = process.exitcode
        if result.get("ok") and result.get("value") != expected_sum:
            result["ok"] = False
            result["error"] = f"unexpected all_reduce value {result.get('value')}"
        results.append(result)

    return {
        "ok": all(result.get("ok") for result in results),
        "world_size": world_size,
        "expected_sum": expected_sum,
        "init_method": init_method,
        "timed_out_ranks": timed_out,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe the NCCL and Gloo group sequence used by vLLM TP.")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.world_size < 2:
        parser.error("world size must be at least two")

    summary = run_probe(args.world_size, args.timeout_s)
    for result in summary["results"]:
        status = "PASS" if result.get("ok") else "FAIL"
        print(f"[{status}] rank={result['rank']} "
              f"{json.dumps(result, sort_keys=True, ensure_ascii=False)}",
              flush=True)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False), flush=True)
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True,
                       ensure_ascii=False) + "\n")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
