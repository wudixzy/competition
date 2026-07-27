#!/usr/bin/env python3
"""Fail closed when a service or GPU process survives runner cleanup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any


SCHEMA = "bi100-service-postflight-v1"
API_MARKER = b"vllm.entrypoints.openai.api_server"
WORKER_MARKERS = (
    b"VllmWorkerProcess",
    b"vllm.worker",
    b"multiproc_gpu_executor",
    b"multiproc_worker_utils",
)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _read_comm(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _process_rows(proc_root: Path) -> list[Path]:
    return sorted(
        (
            path for path in proc_root.iterdir()
            if path.is_dir() and re.fullmatch(r"[1-9][0-9]*", path.name)
        ),
        key=lambda path: int(path.name),
    )


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def scan(
    proc_root: Path,
    device_root: Path,
    gpu_indices: tuple[int, ...],
    own_pid: int,
) -> dict[str, Any]:
    expected_devices = {
        (device_root / f"iluvatar{index}").resolve(): index
        for index in gpu_indices
    }
    missing_devices = [
        index for path, index in expected_devices.items() if not path.exists()
    ]
    api_server_pids: list[int] = []
    worker_pids: list[int] = []
    gpu_processes: list[dict[str, Any]] = []
    scan_errors: list[dict[str, Any]] = []

    try:
        process_paths = _process_rows(proc_root)
    except OSError as exc:
        return {
            "schema": SCHEMA,
            "version": 1,
            "qualified": False,
            "gpu_indices": list(gpu_indices),
            "missing_devices": missing_devices,
            "api_server_pids": [],
            "worker_pids": [],
            "gpu_processes": [],
            "scan_errors": [{
                "operation": "list_proc",
                "error": type(exc).__name__,
            }],
        }

    for process_path in process_paths:
        pid = int(process_path.name)
        if pid == own_pid:
            continue
        try:
            command = _read_bytes(process_path / "cmdline")
            comm = _read_comm(process_path / "comm")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as exc:
            scan_errors.append({
                "pid": pid,
                "operation": "read_identity",
                "error": type(exc).__name__,
            })
            continue

        if API_MARKER in command:
            api_server_pids.append(pid)
        if (
            any(marker in command for marker in WORKER_MARKERS)
            or comm.startswith("VllmWorker")
        ):
            worker_pids.append(pid)

        held_devices: set[int] = set()
        fd_root = process_path / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            descriptors = []
        except OSError as exc:
            scan_errors.append({
                "pid": pid,
                "operation": "list_fd",
                "error": type(exc).__name__,
            })
            descriptors = []
        for descriptor in descriptors:
            try:
                target = descriptor.resolve(strict=True)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            except OSError as exc:
                scan_errors.append({
                    "pid": pid,
                    "operation": "resolve_fd",
                    "error": type(exc).__name__,
                })
                continue
            index = expected_devices.get(target)
            if index is not None:
                held_devices.add(index)
        if held_devices:
            gpu_processes.append({
                "pid": pid,
                "comm": comm,
                "gpu_indices": sorted(held_devices),
            })

    qualified = not any((
        missing_devices,
        api_server_pids,
        worker_pids,
        gpu_processes,
        scan_errors,
    ))
    return {
        "schema": SCHEMA,
        "version": 1,
        "qualified": qualified,
        "gpu_indices": list(gpu_indices),
        "missing_devices": missing_devices,
        "api_server_pids": api_server_pids,
        "worker_pids": worker_pids,
        "gpu_processes": gpu_processes,
        "scan_errors": scan_errors,
        "privacy": {
            "command_lines_recorded": False,
            "environment_recorded": False,
        },
    }


def _observation(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "qualified": result.get("qualified") is True,
        "missing_devices": result.get("missing_devices", []),
        "api_server_pids": result.get("api_server_pids", []),
        "worker_pids": result.get("worker_pids", []),
        "gpu_processes": result.get("gpu_processes", []),
        "scan_errors": result.get("scan_errors", []),
    }


def scan_until_stable(
    scan_once: Any,
    *,
    settle_timeout_s: float,
    clean_samples: int,
    sample_interval_s: float,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    if settle_timeout_s < 0:
        raise ValueError("settle_timeout_s must be non-negative")
    if clean_samples < 1:
        raise ValueError("clean_samples must be positive")
    if sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive")
    if settle_timeout_s == 0 and clean_samples != 1:
        raise ValueError(
            "clean_samples must be 1 when settling is disabled")

    started = monotonic()
    deadline = started + settle_timeout_s
    clean_streak = 0
    observations: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    while True:
        final = scan_once()
        observations.append(_observation(final))
        if final.get("qualified") is True:
            clean_streak += 1
        else:
            clean_streak = 0
        if clean_streak >= clean_samples:
            qualified = True
            break
        current = monotonic()
        if current >= deadline:
            qualified = False
            break
        sleep(min(sample_interval_s, max(0.0, deadline - current)))

    assert final is not None
    result = dict(final)
    result["qualified"] = qualified
    result["settling"] = {
        "timeout_s": settle_timeout_s,
        "sample_interval_s": sample_interval_s,
        "required_clean_samples": clean_samples,
        "final_clean_streak": clean_streak,
        "attempts": len(observations),
        "elapsed_s": monotonic() - started,
        "observations": observations,
    }
    return result


def _parse_gpu_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU indices must be comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError(
            "GPU indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("GPU indices must be unique")
    return indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus", type=_parse_gpu_indices, default=(0, 1, 2, 3))
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--device-root", type=Path, default=Path("/dev"))
    parser.add_argument("--settle-timeout-s", type=float, default=0.0)
    parser.add_argument("--clean-samples", type=int, default=1)
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.settle_timeout_s < 0:
        parser.error("--settle-timeout-s must be non-negative")
    if args.clean_samples < 1:
        parser.error("--clean-samples must be positive")
    if args.sample_interval_s <= 0:
        parser.error("--sample-interval-s must be positive")
    if args.settle_timeout_s == 0 and args.clean_samples != 1:
        parser.error(
            "--clean-samples must be 1 when settling is disabled")

    result = scan_until_stable(
        lambda: scan(
            args.proc_root.resolve(),
            args.device_root.resolve(),
            args.gpus,
            os.getpid(),
        ),
        settle_timeout_s=args.settle_timeout_s,
        clean_samples=args.clean_samples,
        sample_interval_s=args.sample_interval_s,
    )
    _atomic_write(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
