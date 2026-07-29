#!/usr/bin/env python3
"""Content-addressed builder for the BI100 fused-prefill extension."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


SCHEMA = "bi100-corex-extension-build-cache-v1"
ARTIFACT_NAME = "corex_fused_paged_prefill_split4.so"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def toolchain_identity(
    compiler: Path,
    python: Path,
) -> dict[str, Any]:
    compiler = compiler.resolve(strict=True)
    python = python.resolve(strict=True)
    torch_probe = _command_output([
        str(python),
        "-c",
        (
            "import json,torch;"
            "print(json.dumps({'torch_version':torch.__version__,"
            "'torch_file':torch.__file__},sort_keys=True))"
        ),
    ])
    return {
        "compiler_path": str(compiler),
        "compiler_sha256": sha256_file(compiler),
        "compiler_version_sha256": hashlib.sha256(
            _command_output([str(compiler), "--version"]).encode("utf-8")
        ).hexdigest(),
        "python_path": str(python),
        "python_sha256": sha256_file(python),
        "torch_probe": json.loads(torch_probe),
        "gpu_arch": "ivcore10",
        "corex_abi": "_GLIBCXX_USE_CXX11_ABI=0",
    }


def cache_key(
    source: Path,
    build_script: Path,
    toolchain: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    inputs = {
        "source_sha256": sha256_file(source),
        "build_script_sha256": sha256_file(build_script),
        "toolchain": toolchain,
        "artifact_name": ARTIFACT_NAME,
    }
    encoded = json.dumps(
        inputs, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest(), inputs


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o755)
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


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


def _valid_cached_entry(
    entry: Path,
    key: str,
    inputs: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    artifact = entry / ARTIFACT_NAME
    manifest_path = entry / "manifest.json"
    if not artifact.is_file() or not manifest_path.is_file():
        return False, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError):
        return False, None
    artifact_sha = sha256_file(artifact)
    valid = (
        manifest.get("schema") == SCHEMA
        and manifest.get("version") == 1
        and manifest.get("cache_key") == key
        and manifest.get("inputs") == inputs
        and manifest.get("artifact", {}).get("sha256") == artifact_sha
        and manifest.get("artifact", {}).get("size_bytes")
        == artifact.stat().st_size
        and manifest.get("build_succeeded") is True
    )
    return valid, manifest if valid else None


def build_or_reuse(
    *,
    source: Path,
    build_script: Path,
    cache_root: Path,
    output: Path,
    compiler: Path,
    python: Path,
    corex_root: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    source = source.resolve(strict=True)
    build_script = build_script.resolve(strict=True)
    expected_source = (
        build_script.parent / "corex_fused_paged_prefill_split4.cu"
    ).resolve(strict=True)
    if source != expected_source:
        raise ValueError(
            "source must be the exact kernel consumed by the build script")
    cache_root = cache_root.resolve()
    output = output.resolve()
    if cache_root == Path("/") or output == Path("/"):
        raise ValueError("cache and output paths must not be filesystem root")
    cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(cache_root, 0o700)
    toolchain = toolchain_identity(compiler, python)
    key, inputs = cache_key(source, build_script, toolchain)
    entry = cache_root / key
    lock_path = cache_root / f"{key}.lock"
    lock_descriptor = os.open(
        lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    cache_hit = False
    manifest: dict[str, Any] | None = None
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        cache_hit, manifest = _valid_cached_entry(entry, key, inputs)
        if not cache_hit:
            if entry.exists():
                shutil.rmtree(entry)
            staging = Path(tempfile.mkdtemp(
                prefix=f".{key}.", dir=cache_root))
            try:
                build_root = staging / "build"
                build_root.mkdir()
                environment = os.environ.copy()
                environment["COREX_ROOT"] = str(corex_root.resolve())
                result = subprocess.run(
                    ["/bin/bash", str(build_script), str(build_root)],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=environment,
                )
                artifact = build_root / ARTIFACT_NAME
                if result.returncode != 0 or not artifact.is_file():
                    raise RuntimeError(
                        "CoreX extension build failed with return code "
                        f"{result.returncode}")
                published_artifact = staging / ARTIFACT_NAME
                artifact.replace(published_artifact)
                artifact_sha = sha256_file(published_artifact)
                manifest = {
                    "schema": SCHEMA,
                    "version": 1,
                    "cache_key": key,
                    "inputs": inputs,
                    "build_succeeded": True,
                    "build_returncode": result.returncode,
                    "artifact": {
                        "name": ARTIFACT_NAME,
                        "sha256": artifact_sha,
                        "size_bytes": published_artifact.stat().st_size,
                    },
                    "stdout_sha256": hashlib.sha256(
                        result.stdout.encode("utf-8")).hexdigest(),
                    "stderr_sha256": hashlib.sha256(
                        result.stderr.encode("utf-8")).hexdigest(),
                    "privacy": {
                        "stdout_persisted": False,
                        "stderr_persisted": False,
                        "credentials_recorded": False,
                    },
                }
                _atomic_json(staging / "manifest.json", manifest)
                staging.replace(entry)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        assert manifest is not None
        _atomic_copy(entry / ARTIFACT_NAME, output)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)

    artifact_sha = sha256_file(output)
    if artifact_sha != manifest["artifact"]["sha256"]:
        raise RuntimeError("published extension differs from cache manifest")
    return {
        "schema": SCHEMA,
        "version": 1,
        "cache_key": key,
        "cache_hit": cache_hit,
        "cache_entry": str(entry),
        "elapsed_s": time.monotonic() - started,
        "artifact": {
            "path": str(output),
            "sha256": artifact_sha,
            "size_bytes": output.stat().st_size,
        },
        "inputs": inputs,
        "authorization": {
            "tp4_service_authorized": False,
            "main_or_yaml_change_authorized": False,
        },
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "qwen3_6_scripts"
        / "corex_fused_paged_prefill_split4.cu",
    )
    parser.add_argument(
        "--build-script",
        type=Path,
        default=root / "qwen3_6_scripts"
        / "build_corex_fused_paged_prefill_split4.sh",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compiler",
        type=Path,
        default=Path("/usr/local/corex-3.2.3/bin/clang++"),
    )
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--corex-root",
        type=Path,
        default=Path("/usr/local/corex-3.2.3"),
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = build_or_reuse(
        source=args.source,
        build_script=args.build_script,
        cache_root=args.cache_root,
        output=args.output,
        compiler=args.compiler,
        python=args.python,
        corex_root=args.corex_root,
    )
    _atomic_json(args.report, value)
    print(json.dumps({
        "cache_key": value["cache_key"],
        "cache_hit": value["cache_hit"],
        "elapsed_s": value["elapsed_s"],
        "artifact_sha256": value["artifact"]["sha256"],
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
