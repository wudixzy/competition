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


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _reference_token(local_logits: Any, rank: int, world_size: int,
                     dist: Any, torch: Any) -> Any:
    gathered = ([torch.empty_like(local_logits) for _ in range(world_size)]
                if rank == 0 else None)
    dist.gather(local_logits, gather_list=gathered, dst=0)
    if rank != 0:
        return None
    return torch.argmax(torch.cat(gathered, dim=-1), dim=-1)


def _sharded_token(local_logits: Any, rank: int, world_size: int,
                   local_vocab_size: int, dist: Any, torch: Any) -> Any:
    local_value, local_index = torch.max(local_logits, dim=-1)
    global_index = local_index + rank * local_vocab_size
    # Every FP16 value and token id below 2**24 is exact in FP32, allowing one
    # two-element gather instead of separate value and index collectives.
    packed = torch.stack((local_value.float(), global_index.float()), dim=-1)
    gathered = ([torch.empty_like(packed) for _ in range(world_size)]
                if rank == 0 else None)
    dist.gather(packed, gather_list=gathered, dst=0)
    if rank != 0:
        return None

    candidates = torch.cat(gathered, dim=0)
    values = candidates[:, 0]
    token_ids = candidates[:, 1].to(torch.long)
    sentinel = torch.full_like(token_ids, torch.iinfo(torch.long).max)

    # torch.argmax returns the first NaN when any NaN is present. Local max
    # retains the first local NaN; selecting the smallest global id preserves
    # the same cross-rank tie-break. Inf and ordinary ties use the same rule.
    nan_mask = torch.isnan(values)
    nan_token = torch.where(nan_mask, token_ids, sentinel).amin()
    global_max = values.amax()
    max_token = torch.where(values == global_max, token_ids, sentinel).amin()
    return torch.where(nan_mask.any(), nan_token, max_token).reshape(1)


def _set_case(local_logits: Any, rank: int, local_vocab_size: int,
              case: str, torch: Any) -> None:
    local_logits.fill_(-17.0)
    points: dict[str, list[tuple[int, int, float]]] = {
        "cross_rank_tie": [(0, 17, 9.0), (3, 3, 9.0)],
        "adjacent_rank_tie": [(1, 100, 8.0), (2, 0, 8.0)],
        "positive_inf": [(0, 200, float("inf")),
                         (2, 1, float("inf"))],
        "nan": [(1, 7, float("nan")), (3, 0, float("nan"))],
    }
    if case == "all_negative_inf":
        local_logits.fill_(float("-inf"))
        return
    for point_rank, index, value in points[case]:
        if rank == point_rank:
            assert 0 <= index < local_vocab_size
            local_logits[0, index] = value


def worker(rank: int, world_size: int, init_method: str, vocab_size: int,
           random_steps: int, warmup: int, iterations: int, repeats: int,
           queue: mp.Queue) -> None:
    setup_corex_env()
    os.environ.update({
        "LOCAL_RANK": str(rank),
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
    })
    try:
        import torch
        import torch.distributed as dist

        if vocab_size % world_size:
            raise ValueError("vocab size must divide evenly across TP ranks")
        local_vocab_size = vocab_size // world_size
        if vocab_size >= 2**24:
            raise ValueError("FP32-packed token ids require vocab_size < 2**24")

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        device = torch.device(f"cuda:{rank}")
        generator = torch.Generator(device=device).manual_seed(20260715 + rank)

        exact_matches = 0
        for _ in range(random_steps):
            logits = torch.randn(
                (1, local_vocab_size), device=device, dtype=torch.float16,
                generator=generator,
            )
            reference = _reference_token(logits, rank, world_size, dist, torch)
            candidate = _sharded_token(
                logits, rank, world_size, local_vocab_size, dist, torch)
            if rank == 0:
                exact_matches += int(torch.equal(reference, candidate))

        edge_results: dict[str, bool] = {}
        edge_cases = [
            "cross_rank_tie",
            "adjacent_rank_tie",
            "positive_inf",
            "nan",
            "all_negative_inf",
        ]
        logits = torch.empty(
            (1, local_vocab_size), device=device, dtype=torch.float16)
        for case in edge_cases:
            _set_case(logits, rank, local_vocab_size, case, torch)
            reference = _reference_token(logits, rank, world_size, dist, torch)
            candidate = _sharded_token(
                logits, rank, world_size, local_vocab_size, dist, torch)
            if rank == 0:
                edge_results[case] = bool(torch.equal(reference, candidate))

        fixed_logits = torch.randn(
            (1, local_vocab_size), device=device, dtype=torch.float16,
            generator=generator,
        )

        def measure(candidate: bool) -> list[float]:
            operation = _sharded_token if candidate else _reference_token
            for _ in range(warmup):
                if candidate:
                    operation(fixed_logits, rank, world_size,
                              local_vocab_size, dist, torch)
                else:
                    operation(fixed_logits, rank, world_size, dist, torch)
            torch.cuda.synchronize()

            trials = []
            for _ in range(repeats):
                dist.barrier()
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(iterations):
                    if candidate:
                        operation(fixed_logits, rank, world_size,
                                  local_vocab_size, dist, torch)
                    else:
                        operation(fixed_logits, rank, world_size, dist, torch)
                torch.cuda.synchronize()
                trials.append(
                    (time.perf_counter() - started) * 1000.0 / iterations)
            return trials

        reference_trials = measure(False)
        candidate_trials = measure(True)
        result: dict[str, Any] = {
            "rank": rank,
            "ok": True,
            "reference": {
                "p10_ms": percentile(reference_trials, 10),
                "median_ms": statistics.median(reference_trials),
                "p90_ms": percentile(reference_trials, 90),
                "trials_ms": reference_trials,
            },
            "candidate": {
                "p10_ms": percentile(candidate_trials, 10),
                "median_ms": statistics.median(candidate_trials),
                "p90_ms": percentile(candidate_trials, 90),
                "trials_ms": candidate_trials,
            },
        }
        if rank == 0:
            result.update({
                "random_exact": exact_matches,
                "random_steps": random_steps,
                "edge_exact": edge_results,
            })
            result["ok"] = (
                exact_matches == random_steps and all(edge_results.values()))
        queue.put(result)
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
            args=(rank, args.world_size, init_method, args.vocab_size,
                  args.random_steps, args.warmup, args.iterations,
                  args.repeats, queue),
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

    aggregate: dict[str, Any] = {}
    if len(results) == args.world_size and all(item.get("ok") for item in results):
        reference_ms = max(
            float(item["reference"]["median_ms"]) for item in results)
        candidate_ms = max(
            float(item["candidate"]["median_ms"]) for item in results)
        saving_ms = reference_ms - candidate_ms
        aggregate = {
            "reference_max_rank_median_ms": reference_ms,
            "candidate_max_rank_median_ms": candidate_ms,
            "saving_ms": saving_ms,
            "speedup": reference_ms / candidate_ms,
            "qualifies_for_implementation": saving_ms >= args.min_saving_ms,
        }

    return {
        "ok": len(results) == args.world_size
              and all(item.get("ok") for item in results),
        "config": {
            "world_size": args.world_size,
            "vocab_size": args.vocab_size,
            "local_vocab_size": args.vocab_size // args.world_size,
            "random_steps": args.random_steps,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "min_saving_ms": args.min_saving_ms,
        },
        "timed_out_ranks": timed_out,
        "aggregate": aggregate,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=248320)
    parser.add_argument("--random-steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--min-saving-ms", type=float, default=0.672)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
