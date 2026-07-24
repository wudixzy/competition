#!/usr/bin/env python3
"""Verify that an active bare-host overlay is bound to the current source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


SCHEMA = "bi100-bare-host-runtime-identity-v1"
VERSION = 1
INSTALL_SCHEMA = "bi100-bare-host-runtime-install-v2"
DIRECT_SOURCE_FILES = {
    "vllm_model": Path("qwen3_6_scripts/qwen3_5.py"),
    "bi100_profile": Path("qwen3_6_scripts/bi100_profile.py"),
    "paged_attention": Path("qwen3_6_scripts/paged_attn.py"),
    "xformers_backend": Path("vllm/attention/backends/xformers.py"),
    "gdn_prefix": Path("qwen3_6_scripts/gdn_prefix.py"),
    "scheduler": Path("qwen3_6_scripts/scheduler.py"),
    "content_cache": Path("vllm/core/block/cpu_kv_content_cache.py"),
    "moe_config": Path(
        "qwen3_6_scripts/qwen3_5_moe/configuration_qwen3_5_moe.py"),
}
GENERATED_FILES = {"block_manager", "cache_trace_outputs"}
REQUIRED_FILES = set(DIRECT_SOURCE_FILES) | GENERATED_FILES


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(
    source_root: Path,
    runtime_site_packages: Path,
    runtime_install: dict[str, Any],
    source_revision: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    source_root = source_root.resolve()
    runtime_site_packages = runtime_site_packages.resolve()

    if (runtime_install.get("schema") != INSTALL_SCHEMA
            or runtime_install.get("version") != 2
            or runtime_install.get("qualified") is not True
            or runtime_install.get("system_site_packages_modified") is not False):
        reasons.append("runtime install report is not qualified")
    if runtime_install.get("source_tree_clean") is not True:
        reasons.append("runtime install was not built from a clean source tree")
    if runtime_install.get("source_revision") != source_revision:
        reasons.append("runtime install revision differs from current source")

    reported_site = runtime_install.get("site_packages")
    if (not isinstance(reported_site, str)
            or Path(reported_site).resolve() != runtime_site_packages):
        reasons.append("active runtime differs from install report")

    versions = runtime_install.get("versions") or {}
    if versions.get("transformers") != "4.55.3":
        reasons.append("runtime Transformers version differs from 4.55.3")

    install_files = runtime_install.get("files")
    if not isinstance(install_files, dict):
        install_files = {}
        reasons.append("runtime install file identities are missing")
    if not REQUIRED_FILES.issubset(install_files):
        reasons.append("runtime install report is missing required files")

    files: dict[str, Any] = {}
    for name in sorted(REQUIRED_FILES):
        row = install_files.get(name)
        if not isinstance(row, dict):
            continue
        source_sha = row.get("source_sha256")
        installed_sha = row.get("installed_sha256")
        installed_path = row.get("installed_path")
        runtime_sha = None
        if isinstance(installed_path, str):
            path = Path(installed_path).resolve()
            if path.is_relative_to(runtime_site_packages) and path.is_file():
                runtime_sha = _digest(path)
            else:
                reasons.append(
                    f"active runtime file is missing or outside overlay: {name}")
        else:
            reasons.append(f"active runtime path is missing: {name}")

        current_source_sha = None
        relative = DIRECT_SOURCE_FILES.get(name)
        if relative is not None:
            source_path = source_root / relative
            if source_path.is_file():
                current_source_sha = _digest(source_path)
            else:
                reasons.append(f"current source file is missing: {name}")
        same = (row.get("same") is True
                and isinstance(source_sha, str)
                and source_sha == installed_sha == runtime_sha
                and (relative is None or current_source_sha == source_sha))
        if not same:
            reasons.append(f"runtime/current source identity differs: {name}")
        files[name] = {
            "generated": relative is None,
            "same": same,
        }

    fixed_sources = {
        "block_manager_base_sha256": source_root
        / "vllm/core/block_manager_v2.py",
        "cache_trace_patcher_sha256": source_root
        / "qwen3_6_scripts/patch_block_manager_cache_trace.py",
        "installer_sha256": source_root
        / "scripts/install_bi100_bare_host_runtime.sh",
    }
    fixed_source_identity = {}
    for field, path in fixed_sources.items():
        current = _digest(path) if path.is_file() else None
        same = current is not None and runtime_install.get(field) == current
        if not same:
            reasons.append(f"runtime install source identity differs: {field}")
        fixed_source_identity[field] = same

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": not reasons,
        "reasons": reasons,
        "source_revision": source_revision,
        "runtime_site_packages": str(runtime_site_packages),
        "files": files,
        "fixed_source_identity": fixed_source_identity,
    }


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runtime-site-packages", type=Path, required=True)
    parser.add_argument("--runtime-install", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    source_revision = subprocess.run(
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
        source_revision,
    )
    _atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
