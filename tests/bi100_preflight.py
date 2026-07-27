#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
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
TERM_GRACE_S = 60.0
KILL_GRACE_S = 20.0
_ACTIVE_CHILD: subprocess.Popen[str] | None = None


class ParentTermination(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _parent_signal_handler(signum: int, _frame: Any) -> None:
    # Ignore repeated termination while probe_gpu gives the child its full
    # graceful shutdown window.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise ParentTermination(signum)


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


def _last_progress_stage(value: str) -> str | None:
    for line in reversed(value.splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        stage = record.get("stage") if isinstance(record, dict) else None
        if isinstance(stage, str) and stage:
            return stage
    return None


def _cleanup_child_group(process: subprocess.Popen[str]) -> bool:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        process.terminate()
    try:
        process.communicate(timeout=TERM_GRACE_S)
        return True
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
        try:
            process.communicate(timeout=KILL_GRACE_S)
            return True
        except subprocess.TimeoutExpired:
            return False


def _cleanup_active_child() -> bool:
    global _ACTIVE_CHILD
    process = _ACTIVE_CHILD
    if process is None:
        return True
    if process.poll() is not None:
        _ACTIVE_CHILD = None
        return True
    cleaned = _cleanup_child_group(process)
    if cleaned:
        _ACTIVE_CHILD = None
    return cleaned


def _spawn_probe_child(
    command: list[str],
    env: dict[str, str],
) -> subprocess.Popen[str]:
    global _ACTIVE_CHILD
    if _ACTIVE_CHILD is not None:
        if _ACTIVE_CHILD.poll() is None:
            raise RuntimeError("previous GPU probe child is still active")
        _ACTIVE_CHILD = None

    blocked = {signal.SIGTERM, signal.SIGINT}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        process = subprocess.Popen(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        _ACTIVE_CHILD = process
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    return process


def probe_gpu(index: int, timeout_s: float, matmul_size: int) -> dict[str, Any]:
    global _ACTIVE_CHILD
    child = textwrap.dedent("""
        import json
        import sys

        index = int(sys.argv[1])
        matmul_size = int(sys.argv[2])

        def progress(stage):
            print(json.dumps({"gpu": index, "stage": stage}), flush=True)

        progress("import_torch")
        import torch

        result = {"gpu": index, "ok": False, "stage": "start"}
        progress("set_device")
        torch.cuda.set_device(index)
        progress("device_info")
        result["device_name"] = torch.cuda.get_device_name(index)
        result["device_capability"] = list(
            torch.cuda.get_device_capability(index))
        progress("mem_get_info")
        result["stage"] = "mem_get_info"
        free, total = torch.cuda.mem_get_info()
        result["free"] = int(free)
        result["total"] = int(total)
        progress("allocate")
        result["stage"] = "allocate"
        a = torch.ones((matmul_size, matmul_size), device=f"cuda:{index}")
        progress("matmul")
        result["stage"] = "matmul"
        b = a @ a
        progress("synchronize")
        result["stage"] = "synchronize"
        torch.cuda.synchronize()
        progress("checksum")
        result["stage"] = "checksum"
        result["checksum"] = float(b.sum().item())
        result["stage"] = "done"
        result["ok"] = True
        print(json.dumps(result, sort_keys=True), flush=True)
    """).strip()
    try:
        process = _spawn_probe_child(
            [sys.executable, "-c", child, str(index), str(matmul_size)],
            corex_env(),
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as initial_timeout:
            stdout = _clean_stream(initial_timeout.stdout)
            stderr = _clean_stream(initial_timeout.stderr)
            termination = "sigterm"
            cleanup_reaped = False
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                termination = f"sigterm_error:{type(error).__name__}"
            try:
                final_stdout, final_stderr = process.communicate(
                    timeout=TERM_GRACE_S)
                stdout = _clean_stream(final_stdout) or stdout
                stderr = _clean_stream(final_stderr) or stderr
                cleanup_reaped = True
            except subprocess.TimeoutExpired as term_timeout:
                stdout = _clean_stream(term_timeout.stdout) or stdout
                stderr = _clean_stream(term_timeout.stderr) or stderr
                termination = "sigkill"
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    termination = f"sigkill_error:{type(error).__name__}"
                try:
                    final_stdout, final_stderr = process.communicate(
                        timeout=KILL_GRACE_S)
                    stdout = _clean_stream(final_stdout) or stdout
                    stderr = _clean_stream(final_stderr) or stderr
                    cleanup_reaped = True
                except subprocess.TimeoutExpired as kill_timeout:
                    stdout = _clean_stream(kill_timeout.stdout) or stdout
                    stderr = _clean_stream(kill_timeout.stderr) or stderr
                    termination = "cleanup_failed"
            if cleanup_reaped:
                _ACTIVE_CHILD = None
            return {
                "gpu": index,
                "ok": False,
                "stage": "timeout",
                "last_progress_stage": _last_progress_stage(stdout),
                "returncode": 124,
                "child_returncode": process.returncode,
                "termination": termination,
                "cleanup_reaped": cleanup_reaped,
                "stdout": stdout,
                "stderr": stderr,
            }

        stdout = stdout.strip()
        stderr = stderr.strip()
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
        parsed["returncode"] = process.returncode
        if stderr:
            parsed["stderr"] = stderr
        if process.returncode != 0:
            parsed["ok"] = False
        _ACTIVE_CHILD = None
        return parsed
    except ParentTermination:
        _cleanup_active_child()
        raise


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

    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in previous_handlers:
        signal.signal(signum, _parent_signal_handler)
    try:
        results = [
            probe_gpu(index, args.timeout_s, args.matmul_size)
            for index in args.gpus
        ]
    except ParentTermination as termination:
        cleanup_ok = _cleanup_active_child()
        print(
            f"GPU preflight interrupted by signal {termination.signum}; "
            f"child_cleanup_ok={str(cleanup_ok).lower()}",
            file=sys.stderr,
            flush=True,
        )
        return 128 + termination.signum
    finally:
        _cleanup_active_child()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    for result in results:
        status = "PASS" if result.get("ok") else "FAIL"
        detail = json.dumps(result, sort_keys=True, ensure_ascii=False)
        print(f"[{status}] gpu={result.get('gpu')} {detail}", flush=True)

    summary = {
        "schema": "bi100-gpu-preflight-v1",
        "version": 1,
        "gpus": args.gpus,
        "matmul_size": args.matmul_size,
        "timeout_s": args.timeout_s,
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
