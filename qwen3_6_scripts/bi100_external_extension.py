"""Load a hash-bound private experiment extension when explicitly requested."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hashed_private_extension(
    module_name: str,
    *,
    path_environment: str,
    sha256_environment: str,
    required_callable: str,
) -> ModuleType | None:
    """Return an exact private artifact, or None when no override is set."""
    raw_path = os.environ.get(path_environment, "").strip()
    expected_sha256 = os.environ.get(sha256_environment, "").strip()
    if bool(raw_path) != bool(expected_sha256):
        raise RuntimeError(
            "external extension path and SHA-256 must be set together")
    if not raw_path:
        return None
    if (
        len(expected_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_sha256
        )
    ):
        raise RuntimeError("external extension SHA-256 is invalid")

    path = Path(raw_path).resolve(strict=True)
    if (
        not path.is_file()
        or not path.is_relative_to(Path("/tmp"))
        or path.stat().st_size <= 0
        or path.stat().st_mode & 0o022
    ):
        raise RuntimeError(
            "external extension must be a non-writable file under /tmp")
    if _sha256(path) != expected_sha256:
        raise RuntimeError("external extension SHA-256 mismatch")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create external extension loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, required_callable, None)):
        raise RuntimeError("external extension callable is missing")
    return module
