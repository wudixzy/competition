#!/usr/bin/env python3
"""Build an attested runtime contract for one private quality run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import quality_runtime_contract as runtime_contract
import verify_bare_host_runtime_identity as runtime_identity


Json = dict[str, Any]


def service_command(model_path: str) -> list[str]:
    return runtime_contract.service_command(model_path)


def service_environment(
    runtime_site_packages: str,
    *,
    gdn_cache_policy: str,
    gdn_restore_mode: str,
    fused_prefill: str,
    kv_eviction_policy: str,
    kernel_profile: str = "submission",
) -> dict[str, str]:
    return runtime_contract.service_environment(
        runtime_site_packages,
        gdn_cache_policy=gdn_cache_policy,
        gdn_restore_mode=gdn_restore_mode,
        fused_prefill=fused_prefill,
        kv_eviction_policy=kv_eviction_policy,
        kernel_profile=kernel_profile,
    )


def build_contract(
    *,
    source_revision: str,
    runtime_overlay_sha256: str,
    runtime_site_packages: str,
    model_path: str,
    instance: str,
    optimization_label: str,
    gdn_cache_policy: str,
    gdn_restore_mode: str,
    fused_prefill: str,
    kv_eviction_policy: str,
    kernel_profile: str = "submission",
) -> Json:
    identity = f"bare-host-overlay-v1:{runtime_overlay_sha256[:20]}"
    return {
        "schema": "bi100-quality-runtime-contract-v1",
        "version": 1,
        "source_revision": source_revision,
        "runtime_identity": identity,
        "runtime_overlay_sha256": runtime_overlay_sha256,
        "instance": instance,
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": model_path,
        "tokenizer_path": model_path,
        "served_model_name": "llm",
        "base_image": runtime_contract.BASE_IMAGE,
        "command": service_command(model_path),
        "environment": service_environment(
            runtime_site_packages,
            gdn_cache_policy=gdn_cache_policy,
            gdn_restore_mode=gdn_restore_mode,
            fused_prefill=fused_prefill,
            kv_eviction_policy=kv_eviction_policy,
            kernel_profile=kernel_profile,
        ),
        "cache_trace_enabled": True,
        "optimization_label": optimization_label,
    }


def _atomic_write(path: Path, value: Json) -> None:
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
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--optimization-label", required=True)
    parser.add_argument(
        "--gdn-cache-policy", choices=("fine32", "admission64"),
        required=True)
    parser.add_argument(
        "--gdn-restore-mode",
        choices=("direct", "hybrid64", "aligned"),
        required=True,
    )
    parser.add_argument("--fused-prefill", choices=("0", "1"), required=True)
    parser.add_argument(
        "--kv-eviction-policy", choices=("lru", "frequency"), required=True)
    parser.add_argument(
        "--kernel-profile",
        choices=tuple(runtime_contract.KERNEL_PROFILES),
        default="submission",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    runtime_site_packages = args.runtime_site_packages.resolve()
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        parser.error(f"model path is missing: {model_path}")
    source_revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    install = json.loads(args.runtime_install.read_text(encoding="utf-8"))
    identity_report = runtime_identity.verify(
        source_root,
        runtime_site_packages,
        install,
        source_revision,
    )
    if identity_report["qualified"] is not True:
        print(json.dumps(identity_report, indent=2, sort_keys=True))
        return 1
    overlay_sha = identity_report.get("runtime_tree_sha256")
    if not runtime_contract.is_sha256(overlay_sha):
        raise RuntimeError("qualified runtime identity lacks a tree SHA-256")

    contract = build_contract(
        source_revision=source_revision,
        runtime_overlay_sha256=overlay_sha,
        runtime_site_packages=str(runtime_site_packages),
        model_path=str(model_path),
        instance=args.instance,
        optimization_label=args.optimization_label,
        gdn_cache_policy=args.gdn_cache_policy,
        gdn_restore_mode=args.gdn_restore_mode,
        fused_prefill=args.fused_prefill,
        kv_eviction_policy=args.kv_eviction_policy,
        kernel_profile=args.kernel_profile,
    )
    expected = {
        "source_revision": source_revision,
        "runtime_identity": contract["runtime_identity"],
        "instance": args.instance,
        "gpu_count": 4,
        "tensor_parallel_size": 4,
        "max_model_len": 262144,
        "model_path": str(model_path),
        "tokenizer_path": str(model_path),
        "served_model_name": "llm",
    }
    contract_sha = runtime_contract.validate_runtime_contract(
        contract, expected, require_cache_trace=True)
    _atomic_write(args.out, contract)
    print(json.dumps({
        "contract_sha256": contract_sha,
        "out": str(args.out),
        "qualified": True,
        "runtime_identity": contract["runtime_identity"],
        "runtime_overlay_sha256": overlay_sha,
        "source_revision": source_revision,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
