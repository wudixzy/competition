#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "bi100-m1-48-runtime-identity-v1"
VERSION = 1
RUNTIME_SCHEMA = "bi100-bare-host-runtime-install-v2"
SOURCE_FILES = {
    "vllm_model": Path("qwen3_6_scripts/qwen3_5.py"),
    "bi100_profile": Path("qwen3_6_scripts/bi100_profile.py"),
    "paged_attention": Path("qwen3_6_scripts/paged_attn.py"),
    "xformers_backend": Path("vllm/attention/backends/xformers.py"),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(
    source_root: Path,
    runtime_site_packages: Path,
    runtime_install: dict[str, Any],
    source_revision: str,
) -> dict[str, Any]:
    reasons = []
    runtime_site_packages = runtime_site_packages.resolve()
    if (runtime_install.get("schema") != RUNTIME_SCHEMA
            or runtime_install.get("version") != 2
            or runtime_install.get("qualified") is not True):
        reasons.append("runtime install report is not qualified")
    reported_site_packages = runtime_install.get("site_packages")
    if (not isinstance(reported_site_packages, str)
            or Path(reported_site_packages).resolve()
            != runtime_site_packages):
        reasons.append("active runtime differs from install report")
    install_files = runtime_install.get("files")
    if not isinstance(install_files, dict):
        install_files = {}
        reasons.append("runtime install file identities are missing")

    files = {}
    for name, relative in SOURCE_FILES.items():
        path = source_root / relative
        if not path.is_file():
            reasons.append(f"current source file is missing: {name}")
            continue
        current = _digest(path)
        installed = install_files.get(name) or {}
        install_source = installed.get("source_sha256")
        install_target = installed.get("installed_sha256")
        runtime_target = installed.get("installed_path")
        runtime_digest = None
        if isinstance(runtime_target, str):
            runtime_path = Path(runtime_target).resolve()
            if (runtime_path.is_relative_to(runtime_site_packages)
                    and runtime_path.is_file()):
                runtime_digest = _digest(runtime_path)
            else:
                reasons.append(
                    f"active runtime file is missing or outside overlay: {name}")
        else:
            reasons.append(f"active runtime path is missing: {name}")
        same = (installed.get("same") is True
                and current == install_source == install_target
                == runtime_digest)
        if not same:
            reasons.append(f"runtime/current source identity differs: {name}")
        files[name] = {
            "relative_path": relative.as_posix(),
            "current_source_sha256": current,
            "install_source_sha256": install_source,
            "installed_sha256": install_target,
            "runtime_installed_sha256": runtime_digest,
            "same": same,
        }
    worker_path = runtime_site_packages / "vllm" / "worker" / "worker.py"
    runtime_worker_sha256 = None
    startup_profile_guard_patch = False
    if worker_path.is_file():
        runtime_worker_sha256 = _digest(worker_path)
        worker_text = worker_path.read_text(encoding="utf-8")
        startup_profile_guard_patch = (
            "Mark this synthetic pass so BI100_PROFILE can exclude"
            in worker_text
            and 'os.environ["BI100_IN_STARTUP_PROFILE"] = "1"'
            in worker_text
        )
    else:
        reasons.append("active runtime worker is missing")
    install_worker_sha256 = runtime_install.get("worker_sha256")
    if (not startup_profile_guard_patch
            or runtime_install.get("startup_profile_guard_patch") is not True
            or runtime_worker_sha256 != install_worker_sha256):
        reasons.append("active runtime startup-profile guard differs")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "source_revision": source_revision.strip(),
        "runtime_site_packages": str(runtime_site_packages),
        "install_worker_sha256": install_worker_sha256,
        "runtime_worker_sha256": runtime_worker_sha256,
        "startup_profile_guard_patch": startup_profile_guard_patch,
        "files": files,
    }


def _write_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-site-packages", type=Path, required=True)
    parser.add_argument("--runtime-install", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime_install = json.loads(
        args.runtime_install.read_text(encoding="utf-8"))
    report = verify(
        source_root,
        args.runtime_site_packages,
        runtime_install,
        revision,
    )
    _write_atomic(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
