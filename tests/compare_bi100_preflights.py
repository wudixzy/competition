#!/usr/bin/env python3
"""Compare BI100 preflights across isolated service lifetimes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any


SOURCE_SCHEMA = "bi100-gpu-preflight-v1"
SCHEMA = "bi100-gpu-preflight-comparison-v1"
VERSION = 1
EXPECTED_GPUS = [0, 1, 2, 3]
LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _positive_int(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and value > 0)


def _validate_stage(
    label: str,
    value: Any,
    reasons: list[str],
) -> tuple[dict[int, tuple[Any, ...]], dict[str, Any]]:
    signatures: dict[int, tuple[Any, ...]] = {}
    safe_summary: dict[str, Any] = {"label": label, "qualified": False}
    if not isinstance(value, dict):
        reasons.append(f"{label} preflight must be an object")
        return signatures, safe_summary
    if value.get("schema") != SOURCE_SCHEMA or value.get("version") != VERSION:
        reasons.append(f"{label} preflight schema is invalid")
    if value.get("ok") is not True:
        reasons.append(f"{label} preflight is not qualified")
    if value.get("gpus") != EXPECTED_GPUS:
        reasons.append(
            f"{label} GPU order must equal {EXPECTED_GPUS}, "
            f"got {value.get('gpus')!r}")
    matmul_size = value.get("matmul_size")
    if not _positive_int(matmul_size):
        reasons.append(f"{label} matmul_size is invalid")
        matmul_size = None
    timeout_s = value.get("timeout_s")
    if (not isinstance(timeout_s, (int, float))
            or isinstance(timeout_s, bool) or not math.isfinite(timeout_s)
            or timeout_s <= 0):
        reasons.append(f"{label} timeout_s is invalid")
        timeout_s = None
    results = value.get("results")
    if not isinstance(results, list) or len(results) != len(EXPECTED_GPUS):
        reasons.append(f"{label} must contain exactly four GPU results")
        results = []

    safe_results = []
    for expected_gpu, result in zip(EXPECTED_GPUS, results):
        if not isinstance(result, dict):
            reasons.append(f"{label} GPU {expected_gpu} result must be an object")
            continue
        gpu = result.get("gpu")
        name = result.get("device_name")
        capability = result.get("device_capability")
        free = result.get("free")
        total = result.get("total")
        checksum = result.get("checksum")
        if gpu != expected_gpu:
            reasons.append(
                f"{label} GPU result order differs at {expected_gpu}: {gpu!r}")
        if result.get("ok") is not True or result.get("stage") != "done" \
                or result.get("returncode") != 0:
            reasons.append(f"{label} GPU {expected_gpu} health probe failed")
        if not isinstance(name, str) or not name:
            reasons.append(f"{label} GPU {expected_gpu} device name is invalid")
        if (not isinstance(capability, list) or len(capability) != 2
                or not all(isinstance(item, int) and not isinstance(item, bool)
                           and item >= 0 for item in capability)):
            reasons.append(
                f"{label} GPU {expected_gpu} capability is invalid")
        if not _positive_int(total):
            reasons.append(f"{label} GPU {expected_gpu} total memory is invalid")
        if (not _positive_int(free) or not _positive_int(total)
                or free > total):
            reasons.append(f"{label} GPU {expected_gpu} free memory is invalid")
        expected_checksum = matmul_size ** 3 if matmul_size is not None else None
        if (not isinstance(checksum, (int, float))
                or isinstance(checksum, bool) or not math.isfinite(checksum)
                or expected_checksum is None
                or not math.isclose(checksum, expected_checksum,
                                    rel_tol=0.0, abs_tol=0.5)):
            reasons.append(
                f"{label} GPU {expected_gpu} matmul checksum is invalid")
        signatures[expected_gpu] = (
            name,
            tuple(capability) if isinstance(capability, list) else capability,
            total,
            checksum,
            matmul_size,
            timeout_s,
        )
        safe_results.append({
            "gpu": gpu,
            "device_name": name,
            "device_capability": capability,
            "free": free,
            "total": total,
            "checksum": checksum,
            "ok": result.get("ok"),
        })
    safe_summary.update({
        "qualified": not any(reason.startswith(f"{label} ")
                             for reason in reasons),
        "matmul_size": matmul_size,
        "timeout_s": timeout_s,
        "results": safe_results,
    })
    return signatures, safe_summary


def compare(
    stages: list[tuple[str, Any]],
    *,
    max_free_memory_drop_bytes: int | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if (max_free_memory_drop_bytes is not None
            and (not isinstance(max_free_memory_drop_bytes, int)
                 or isinstance(max_free_memory_drop_bytes, bool)
                 or max_free_memory_drop_bytes < 0)):
        reasons.append("max_free_memory_drop_bytes must be non-negative")
        max_free_memory_drop_bytes = None
    if len(stages) < 2:
        reasons.append("at least two preflight stages are required")
    labels = [label for label, _ in stages]
    if len(set(labels)) != len(labels):
        reasons.append("preflight stage labels must be unique")

    baseline: dict[int, tuple[Any, ...]] | None = None
    baseline_free: dict[int, int] | None = None
    summaries = []
    for label, value in stages:
        signatures, summary = _validate_stage(label, value, reasons)
        summaries.append(summary)
        current_free = {
            result["gpu"]: result["free"]
            for result in summary.get("results", [])
            if isinstance(result, dict)
            and isinstance(result.get("gpu"), int)
            and _positive_int(result.get("free"))
        }
        if baseline is None:
            baseline = signatures
            baseline_free = current_free
        elif signatures != baseline:
            reasons.append(
                f"{label} GPU topology or deterministic result differs "
                "from the first preflight")
        if (baseline_free is not None
                and max_free_memory_drop_bytes is not None
                and current_free):
            drops = {
                gpu: max(0, baseline_free[gpu] - current_free[gpu])
                for gpu in sorted(set(baseline_free) & set(current_free))
            }
            summary["free_memory_drop_from_first_bytes"] = {
                str(gpu): drop for gpu, drop in drops.items()
            }
            leaked = {
                gpu: drop for gpu, drop in drops.items()
                if drop > max_free_memory_drop_bytes
            }
            if leaked:
                reasons.append(
                    f"{label} free-memory drop exceeds "
                    f"{max_free_memory_drop_bytes} bytes: {leaked}")
                summary["qualified"] = False
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "max_free_memory_drop_bytes": max_free_memory_drop_bytes,
        "stages": summaries,
    }


def _parse_stage(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--preflight must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError(f"invalid preflight label: {label!r}")
    if not raw_path:
        raise argparse.ArgumentTypeError("preflight path must not be empty")
    return label, Path(raw_path)


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2,
                      sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight", action="append", type=_parse_stage, required=True)
    parser.add_argument("--max-free-memory-drop-bytes", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        stages = [
            (label, json.loads(path.read_text(encoding="utf-8")))
            for label, path in args.preflight
        ]
        report = compare(
            stages,
            max_free_memory_drop_bytes=args.max_free_memory_drop_bytes,
        )
    except Exception as error:
        report = {
            "schema": SCHEMA,
            "version": VERSION,
            "qualified": False,
            "reasons": [f"cannot compare GPU preflights: {error}"],
        }
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
