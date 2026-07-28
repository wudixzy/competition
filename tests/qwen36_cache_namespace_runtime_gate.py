#!/usr/bin/env python3
"""Validate multimodal cache namespaces in an installed BI100 overlay."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch


SCHEMA = "qwen36-cache-namespace-runtime-gate-v2"
VERSION = 2
REQUIRED_CHECKS = (
    "module_bound_to_overlay",
    "same_palette_stable",
    "different_palette_isolated",
    "different_transparency_isolated",
    "empty_multimodal_matches_text",
    "truthiness_not_evaluated",
    "normalization_error_is_request_local",
    "release_clears_request_state",
    "request_id_reuse_gets_fresh_namespace",
)


def _valid_revision(value: str) -> bool:
    return (
        len(value) in (40, 64)
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _new_manager(manager_class: type[Any]) -> Any:
    manager = manager_class.__new__(manager_class)
    manager.block_size = 16
    manager._runtime_cache_namespace = hashlib.sha256(
        b"m1-89-runtime-gate").digest()
    manager._request_local_namespace = {}
    manager._warned_mm_namespace_requests = set()
    return manager


def _palette_image(
    image_module: Any,
    *,
    red: int,
    transparency: int,
) -> Any:
    image = image_module.new("P", (2, 2))
    image.putdata([0, 1, 0, 1])
    palette = [0] * 768
    palette[0:3] = [red, 20, 30]
    palette[3:6] = [40, 50, 60]
    image.putpalette(palette)
    image.info["transparency"] = transparency
    return image


def qualify_checks(
    checks: dict[str, bool],
    errors: dict[str, str],
) -> tuple[bool, list[str]]:
    reasons = []
    if tuple(checks) != REQUIRED_CHECKS:
        reasons.append("runtime check order or identity differs")
    for name in REQUIRED_CHECKS:
        if checks.get(name) is not True:
            reasons.append(f"runtime check failed: {name}")
    if errors:
        reasons.append("one or more runtime checks raised an exception")
    return not reasons, reasons


def run_gate(
    runtime_site_packages: Path,
    source_revision: str,
) -> dict[str, Any]:
    if not _valid_revision(source_revision):
        raise ValueError("source revision must be a full lowercase Git digest")
    runtime_site_packages = runtime_site_packages.resolve(strict=True)
    checks: dict[str, bool] = {}
    errors: dict[str, str] = {}

    def check(name: str, operation: Callable[[], bool]) -> None:
        try:
            checks[name] = operation() is True
        except Exception as exc:
            checks[name] = False
            errors[name] = type(exc).__name__

    block_manager_module = importlib.import_module(
        "vllm.core.block_manager_v2")
    sequence_module = importlib.import_module("vllm.sequence")
    image_module = importlib.import_module("PIL.Image")
    manager_class = block_manager_module.BlockSpaceManagerV2
    module_path = Path(block_manager_module.__file__).resolve(strict=True)
    sequence_module_path = Path(
        sequence_module.__file__).resolve(strict=True)
    group = SimpleNamespace(
        lora_request=None,
        prompt_adapter_request=None,
    )

    check(
        "module_bound_to_overlay",
        lambda: (
            module_path.is_relative_to(runtime_site_packages)
            and sequence_module_path.is_relative_to(runtime_site_packages)
        ),
    )

    def palette_checks() -> tuple[bytes, bytes, bytes, bytes]:
        manager = _new_manager(manager_class)
        same_a = _palette_image(
            image_module, red=10, transparency=0)
        same_b = _palette_image(
            image_module, red=10, transparency=0)
        different_palette = _palette_image(
            image_module, red=200, transparency=0)
        different_transparency = _palette_image(
            image_module, red=10, transparency=1)
        return tuple(
            manager._hash_multi_modal_namespace({"image": image})
            for image in (
                same_a,
                same_b,
                different_palette,
                different_transparency,
            )
        )

    palette_hashes: tuple[bytes, ...] = ()

    def get_palette_hashes() -> bool:
        nonlocal palette_hashes
        palette_hashes = palette_checks()
        return len(palette_hashes) == 4

    check("same_palette_stable", lambda: (
        get_palette_hashes()
        and palette_hashes[0] == palette_hashes[1]
    ))
    check("different_palette_isolated", lambda: (
        len(palette_hashes) == 4
        and palette_hashes[0] != palette_hashes[2]
    ))
    check("different_transparency_isolated", lambda: (
        len(palette_hashes) == 4
        and palette_hashes[0] != palette_hashes[3]
    ))

    def empty_multimodal_matches_text() -> bool:
        manager = _new_manager(manager_class)
        runtime_empty = sequence_module.Sequence.multi_modal_data.__get__(
            SimpleNamespace(inputs={}),
            sequence_module.Sequence,
        )
        text = manager._get_cache_namespace(
            SimpleNamespace(multi_modal_data=None), "text", group)
        empty_multimodal = manager._get_cache_namespace(
            SimpleNamespace(multi_modal_data=runtime_empty),
            "empty-mm",
            group,
        )
        return runtime_empty == {} and text == empty_multimodal

    check(
        "empty_multimodal_matches_text",
        empty_multimodal_matches_text,
    )

    def truthiness_not_evaluated() -> bool:
        class Ambiguous:

            def __bool__(self) -> bool:
                raise AssertionError("multimodal truthiness was evaluated")

        manager = _new_manager(manager_class)
        namespace = manager._get_cache_namespace(
            SimpleNamespace(multi_modal_data=Ambiguous()),
            "ambiguous",
            group,
        )
        return (
            len(namespace) == 32
            and "ambiguous" in manager._request_local_namespace
        )

    check("truthiness_not_evaluated", truthiness_not_evaluated)

    def normalization_error_is_request_local() -> bool:
        manager = _new_manager(manager_class)
        image = image_module.new("RGB", (1, 1), color=(1, 2, 3))
        sequence = SimpleNamespace(multi_modal_data={"image": image})
        with patch.object(
                image_module.Image, "tobytes",
                autospec=True,
                side_effect=OSError("synthetic normalization failure")):
            first = manager._get_cache_namespace(
                sequence, "normalization-error", group)
            second = manager._get_cache_namespace(
                sequence, "normalization-error", group)
        return (
            first == second
            and "normalization-error" in manager._request_local_namespace
        )

    check(
        "normalization_error_is_request_local",
        normalization_error_is_request_local,
    )

    def release_clears_request_state() -> bool:
        manager = _new_manager(manager_class)
        manager._request_local_fallback_cache_namespace("release")
        manager._warned_mm_namespace_requests.add("release")
        manager.release_request_cache_namespace("release")
        return (
            "release" not in manager._request_local_namespace
            and "release" not in manager._warned_mm_namespace_requests
        )

    check("release_clears_request_state", release_clears_request_state)

    def request_id_reuse_gets_fresh_namespace() -> bool:
        manager = _new_manager(manager_class)
        with patch.object(
                os, "urandom", side_effect=[b"a" * 32, b"b" * 32]):
            first = manager._request_local_fallback_cache_namespace("reuse")
            manager.release_request_cache_namespace("reuse")
            second = manager._request_local_fallback_cache_namespace("reuse")
        return first != second

    check(
        "request_id_reuse_gets_fresh_namespace",
        request_id_reuse_gets_fresh_namespace,
    )

    qualified, reasons = qualify_checks(checks, errors)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": qualified,
        "reasons": reasons,
        "source_revision": source_revision,
        "runtime_site_packages": str(runtime_site_packages),
        "block_manager_module_sha256": _sha256(module_path),
        "sequence_module_sha256": _sha256(sequence_module_path),
        "pillow_version": importlib.metadata.version("Pillow"),
        "checks": checks,
        "error_types": errors,
        "privacy": {
            "contains_image_bytes": False,
            "contains_namespace_digest": False,
            "contains_request_id": False,
            "contains_prompt_or_output": False,
            "contains_credentials": False,
        },
        "gpu_execution_required": False,
        "model_execution_performed": False,
        "production_promotion_authorized": False,
    }


def failure_report(
    runtime_site_packages: Path,
    source_revision: str,
    error_type: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "qualified": False,
        "reasons": ["runtime gate initialization failed"],
        "source_revision": (
            source_revision if _valid_revision(source_revision) else None),
        "runtime_site_packages": str(runtime_site_packages),
        "checks": {},
        "error_types": {"initialization": error_type},
        "privacy": {
            "contains_image_bytes": False,
            "contains_namespace_digest": False,
            "contains_request_id": False,
            "contains_prompt_or_output": False,
            "contains_credentials": False,
            "contains_exception_message": False,
        },
        "gpu_execution_required": False,
        "model_execution_performed": False,
        "production_promotion_authorized": False,
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-site-packages", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_gate(
            args.runtime_site_packages,
            args.source_revision,
        )
    except Exception as exc:
        report = failure_report(
            args.runtime_site_packages,
            args.source_revision,
            type(exc).__name__,
        )
    _atomic_write(args.out, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
