#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
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


def _prepend_env_list(env: dict[str, str], key: str, values: list[str]) -> None:
    existing = env.get(key, "")
    parts = values + ([existing] if existing else [])
    env[key] = ":".join(parts)


def corex_env() -> dict[str, str]:
    env = os.environ.copy()
    _prepend_env_list(env, "PATH", COREX_BIN_PATHS)
    _prepend_env_list(env, "LD_LIBRARY_PATH", COREX_LIBRARY_PATHS)
    _prepend_env_list(env, "PYTHONPATH", COREX_PYTHON_PATHS)
    return env


def _clean_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip()
    return value.strip()


def probe_gpu(index: int, timeout_s: float, matmul_size: int) -> dict[str, Any]:
    child = textwrap.dedent("""
        import json
        import sys
        import torch

        index = int(sys.argv[1])
        matmul_size = int(sys.argv[2])
        result = {"gpu": index, "ok": False, "stage": "start"}
        torch.cuda.set_device(index)
        result["stage"] = "mem_get_info"
        free, total = torch.cuda.mem_get_info()
        result["free"] = int(free)
        result["total"] = int(total)
        result["stage"] = "matmul"
        a = torch.ones((matmul_size, matmul_size), device=f"cuda:{index}")
        b = a @ a
        torch.cuda.synchronize()
        result["checksum"] = float(b.sum().item())
        result["stage"] = "done"
        result["ok"] = True
        print(json.dumps(result, sort_keys=True))
    """).strip()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", child, str(index), str(matmul_size)],
            env=corex_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "gpu": index,
            "ok": False,
            "stage": "timeout",
            "returncode": 124,
            "stdout": _clean_stream(exc.stdout),
            "stderr": _clean_stream(exc.stderr),
        }

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    parsed: dict[str, Any] | None = None
    if stdout:
        last_line = stdout.splitlines()[-1]
        try:
            parsed = json.loads(last_line)
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        parsed = {
            "gpu": index,
            "ok": False,
            "stage": "parse_output",
        }
    parsed["returncode"] = completed.returncode
    if stderr:
        parsed["stderr"] = stderr
    if completed.returncode != 0:
        parsed["ok"] = False
    return parsed


def parse_gpus(value: str) -> list[int]:
    gpus = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        gpus.append(int(part))
    if not gpus:
        raise argparse.ArgumentTypeError("at least one GPU index is required")
    return gpus


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe BI100 GPUs before launching TP=4 vLLM.")
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    parser.add_argument("--timeout-s", type=float, default=25.0)
    parser.add_argument("--matmul-size", type=int, default=1024)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    results = [
        probe_gpu(index, args.timeout_s, args.matmul_size)
        for index in args.gpus
    ]
    for result in results:
        status = "PASS" if result.get("ok") else "FAIL"
        detail = json.dumps(result, sort_keys=True, ensure_ascii=False)
        print(f"[{status}] gpu={result.get('gpu')} {detail}", flush=True)

    summary = {
        "ok": all(result.get("ok") for result in results),
        "results": results,
    }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False), flush=True)
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
