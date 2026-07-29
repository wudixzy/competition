#!/usr/bin/env python3
"""Build once or reuse an exact-commit immutable BI100 runtime overlay."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any


SCHEMA = "bi100-runtime-overlay-cache-v1"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_identity(root: Path) -> tuple[str, bool]:
    revision = _git(root, "rev-parse", "HEAD")
    status = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)bench_runs/**",
    )
    return revision, not status


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2,
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


def _verify(
    root: Path,
    overlay: Path,
    output: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(root / "tests" / "verify_bare_host_runtime_identity.py"),
        "--source-root",
        str(root),
        "--runtime-site-packages",
        str(overlay / "site-packages"),
        "--runtime-install",
        str(overlay / "install.json"),
        "--out",
        str(output),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{root / 'tests'}:{overlay / 'site-packages'}:"
        "/usr/local/corex/lib64/python3/dist-packages:"
        "/usr/local/corex/lib/python3/dist-packages"
    )
    environment["LD_LIBRARY_PATH"] = (
        "/usr/local/corex/lib:/usr/local/corex/lib64:"
        "/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:"
        "/usr/local/openmpi/lib"
    )
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "cached runtime overlay failed exact-source verification")
    value = json.loads(output.read_text(encoding="ascii"))
    if value.get("qualified") is not True:
        raise RuntimeError("runtime overlay identity is not qualified")
    return value


def ensure(
    *,
    root: Path,
    cache_root: Path,
    verification_out: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    root = root.resolve(strict=True)
    revision, clean = source_identity(root)
    if not clean:
        raise RuntimeError("runtime overlay cache requires a clean source tree")
    cache_root = cache_root.resolve()
    cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(cache_root, 0o700)
    overlay = cache_root / revision
    lock_descriptor = os.open(
        cache_root / f"{revision}.lock",
        os.O_WRONLY | os.O_CREAT,
        0o600,
    )
    cache_hit = overlay.is_dir()
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        cache_hit = overlay.is_dir()
        if not cache_hit:
            environment = os.environ.copy()
            environment["BI100_BARE_HOST_RUNTIME_ROOT"] = str(overlay)
            result = subprocess.run(
                [
                    str(root / "scripts"
                        / "install_bi100_bare_host_runtime.sh"),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                timeout=1800,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "runtime overlay install failed with return code "
                    f"{result.returncode}")
        identity = _verify(root, overlay, verification_out)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
    return {
        "schema": SCHEMA,
        "version": 1,
        "source_revision": revision,
        "source_tree_clean": clean,
        "cache_hit": cache_hit,
        "elapsed_s": time.monotonic() - started,
        "runtime_root": str(overlay),
        "runtime_site_packages": str(overlay / "site-packages"),
        "runtime_install_report": str(overlay / "install.json"),
        "runtime_tree_sha256": identity["runtime_tree_sha256"],
        "qualified": True,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=root)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--verification-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = ensure(
        root=args.source_root,
        cache_root=args.cache_root,
        verification_out=args.verification_out,
    )
    _atomic_json(args.report, value)
    print(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
