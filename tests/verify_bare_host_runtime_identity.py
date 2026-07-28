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
    "api_server": Path("qwen3_6_scripts/api_server.py"),
    "bi100_env": Path("qwen3_6_scripts/bi100_env.py"),
    "vllm_model": Path("qwen3_6_scripts/qwen3_5.py"),
    "bi100_profile": Path("qwen3_6_scripts/bi100_profile.py"),
    "block_major_kv_cache": Path(
        "qwen3_6_scripts/block_major_kv_cache.py"),
    "block_major_kv_extension": Path(
        "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/"
        "corex_block_major_kv_transfer.so"),
    "fused_paged_prefill_extension": Path(
        "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/"
        "corex_fused_paged_prefill.so"),
    "block_table": Path("vllm/core/block/block_table.py"),
    "chat_utils": Path("qwen3_6_scripts/chat_utils.py"),
    "cli_args": Path("qwen3_6_scripts/cli_args.py"),
    "cpu_gpu_block_allocator": Path(
        "vllm/core/block/cpu_gpu_block_allocator.py"),
    "evictor": Path("vllm/core/evictor_v2.py"),
    "paged_attention": Path("qwen3_6_scripts/paged_attn.py"),
    "xformers_backend": Path("vllm/attention/backends/xformers.py"),
    "gdn_prefix": Path("qwen3_6_scripts/gdn_prefix.py"),
    "mamba_cache": Path("qwen3_6_scripts/mamba_cache.py"),
    "prefix_caching_block": Path(
        "vllm/core/block/prefix_caching_block.py"),
    "protocol": Path("qwen3_6_scripts/protocol.py"),
    "reasoning_abs": Path(
        "qwen3_6_scripts/reasoning/abs_reasoning_parsers.py"),
    "reasoning_init": Path("qwen3_6_scripts/reasoning/__init__.py"),
    "reasoning_qwen3": Path(
        "qwen3_6_scripts/reasoning/qwen3_reasoning_parser.py"),
    "scheduler": Path("qwen3_6_scripts/scheduler.py"),
    "sequence": Path("qwen3_6_scripts/sequence.py"),
    "serving_chat": Path("qwen3_6_scripts/serving_chat.py"),
    "content_cache": Path("vllm/core/block/cpu_kv_content_cache.py"),
    "tool_parser": Path("qwen3_6_scripts/qwen3coder_tool_parser.py"),
    "transformers_qwen3_5_config": Path(
        "qwen3_6_scripts/qwen3_5/configuration_qwen3_5.py"),
    "transformers_qwen3_5_init": Path(
        "qwen3_6_scripts/qwen3_5/__init__.py"),
    "moe_config": Path(
        "qwen3_6_scripts/qwen3_5_moe/configuration_qwen3_5_moe.py"),
    "transformers_qwen3_5_moe_init": Path(
        "qwen3_6_scripts/qwen3_5_moe/__init__.py"),
}
GENERATED_FILES = {
    "block_manager", "cache_engine", "cache_trace_outputs", "model_runner",
    "worker",
}
REQUIRED_FILES = set(DIRECT_SOURCE_FILES) | GENERATED_FILES


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_tree_sha256(root: Path) -> str:
    root = root.resolve()
    digest = hashlib.sha256(b"bi100-runtime-tree-v1\0")
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            kind = b"L"
            payload_sha = hashlib.sha256(
                os.readlink(path).encode("utf-8")).digest()
        elif path.is_file():
            kind = b"F"
            file_digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            payload_sha = file_digest.digest()
        else:
            continue
        encoded = relative.as_posix().encode("utf-8")
        digest.update(kind)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(payload_sha)
    return digest.hexdigest()


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
    if runtime_install.get("block_major_cache_engine_patch") is not True:
        reasons.append("runtime install lacks the block-major CacheEngine patch")
    if runtime_install.get("block_major_worker_capacity_patch") is not True:
        reasons.append("runtime install lacks the block-major capacity patch")
    if runtime_install.get("source_revision") != source_revision:
        reasons.append("runtime install revision differs from current source")

    reported_site = runtime_install.get("site_packages")
    if (not isinstance(reported_site, str)
            or Path(reported_site).resolve() != runtime_site_packages):
        reasons.append("active runtime differs from install report")

    reported_tree_sha = runtime_install.get("runtime_tree_sha256")
    actual_tree_sha = (
        runtime_tree_sha256(runtime_site_packages)
        if runtime_site_packages.is_dir() else None)
    if (not isinstance(reported_tree_sha, str)
            or len(reported_tree_sha) != 64
            or reported_tree_sha != actual_tree_sha):
        reasons.append("active runtime tree identity differs from install report")

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
        "offline_metadata_normalizer_sha256": source_root
        / "scripts/normalize_offline_distribution.py",
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
        "runtime_tree_sha256": actual_tree_sha,
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
