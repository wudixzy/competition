#!/usr/bin/env python3
"""Exercise namespace-aware prefix forks against the installed vLLM package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


SCHEMA = "bi100-prefix-namespace-fork-gate-v2"
VERSION = 2
BLOCK_SIZE = 4
BLOCK_TOKEN_IDS = ([1, 2, 3, 4], [5, 6, 7, 8])
NAMESPACE = b"bi100-prefix-fork-gate"
REUSE_OLD_NAMESPACE = b"bi100-prefix-reuse-old"
REUSE_NEW_NAMESPACE = b"bi100-prefix-reuse-new"
REUSE_OLD_TOKENS = [11, 12, 13, 14]
REUSE_NEW_TOKENS = [21, 22, 23, 24]


def atomic_write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
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


def evaluate_case(allocator_factory: Callable[[int], Any],
                  release_order: str) -> dict[str, Any]:
    allocator = allocator_factory(8)
    source = allocator.allocate_immutable_blocks_with_cache_namespace(
        prev_block=None,
        block_token_ids=[list(tokens) for tokens in BLOCK_TOKEN_IDS],
        cache_namespace=NAMESPACE,
    )
    source_hashes = [block.content_hash for block in source]
    forked = allocator.fork(source[-1])
    forked_hashes = [block.content_hash for block in forked]
    namespaces_match = all(
        block.cache_namespace == NAMESPACE for block in forked)
    hashes_match = forked_hashes == source_hashes

    chains = (source, forked) if release_order == "source-first" else (
        forked, source)
    for chain in chains:
        for block in chain:
            allocator.free(block)

    return {
        "release_order": release_order,
        "block_count": len(source),
        "namespaces_match": namespaces_match,
        "hashes_match": hashes_match,
        "source_hashes": [value.hex() for value in source_hashes],
        "forked_hashes": [value.hex() for value in forked_hashes],
        "free_blocks_after_release": allocator.get_num_free_blocks(),
        "total_blocks": allocator.get_num_total_blocks(),
    }


def evaluate_physical_reuse(
        allocator_factory: Callable[[int], Any]) -> dict[str, Any]:
    allocator = allocator_factory(1)
    old_block = allocator.allocate_immutable_block_with_cache_namespace(
        prev_block=None,
        token_ids=list(REUSE_OLD_TOKENS),
        cache_namespace=REUSE_OLD_NAMESPACE,
    )
    old_block_id = old_block.block_id
    old_hash = old_block.content_hash
    allocator.mark_blocks_as_computed([old_block_id])
    old_computed_before_release = allocator.block_is_computed(old_block_id)
    allocator.free(old_block)

    new_block = allocator.allocate_immutable_block_with_cache_namespace(
        prev_block=None,
        token_ids=list(REUSE_NEW_TOKENS),
        cache_namespace=REUSE_NEW_NAMESPACE,
    )
    new_block_id = new_block.block_id
    new_hash = new_block.content_hash
    new_uncomputed_before_mark = not allocator.block_is_computed(new_block_id)
    old_hash_removed = old_hash not in allocator._cached_blocks
    new_hash_registered = (
        allocator._cached_blocks.get(new_hash) == new_block_id)
    allocator.mark_blocks_as_computed([new_block_id])
    new_computed_after_mark = allocator.block_is_computed(new_block_id)
    allocator.free(new_block)

    return {
        "same_physical_block_id": old_block_id == new_block_id,
        "old_hash_sha256": old_hash.hex(),
        "new_hash_sha256": new_hash.hex(),
        "content_hash_changed": old_hash != new_hash,
        "new_namespace_applied": (
            new_block.cache_namespace == REUSE_NEW_NAMESPACE),
        "old_computed_before_release": old_computed_before_release,
        "new_uncomputed_before_mark": new_uncomputed_before_mark,
        "new_computed_after_mark": new_computed_after_mark,
        "old_hash_removed": old_hash_removed,
        "new_hash_registered": new_hash_registered,
        "free_blocks_after_release": allocator.get_num_free_blocks(),
        "total_blocks": allocator.get_num_total_blocks(),
    }


def build_report(allocator_factory: Callable[[int], Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    reasons: list[str] = []
    for release_order in ("source-first", "fork-first"):
        try:
            case = evaluate_case(allocator_factory, release_order)
            cases.append(case)
            if not case["namespaces_match"]:
                reasons.append(f"{release_order}: fork namespace mismatch")
            if not case["hashes_match"]:
                reasons.append(f"{release_order}: fork hash-chain mismatch")
            if case["free_blocks_after_release"] != case["total_blocks"]:
                reasons.append(f"{release_order}: blocks were not releasable")
        except Exception as error:  # Evidence must survive assertion failures.
            reasons.append(
                f"{release_order}: {type(error).__name__}: {error}")

    physical_reuse: dict[str, Any] = {}
    try:
        physical_reuse = evaluate_physical_reuse(allocator_factory)
        required_true = (
            "same_physical_block_id",
            "content_hash_changed",
            "new_namespace_applied",
            "old_computed_before_release",
            "new_uncomputed_before_mark",
            "new_computed_after_mark",
            "old_hash_removed",
            "new_hash_registered",
        )
        for field in required_true:
            if physical_reuse.get(field) is not True:
                reasons.append(f"physical-reuse: {field} is not true")
        if (physical_reuse.get("free_blocks_after_release")
                != physical_reuse.get("total_blocks")):
            reasons.append("physical-reuse: block was not releasable")
    except Exception as error:
        reasons.append(f"physical-reuse: {type(error).__name__}: {error}")

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "namespace_sha256": hashlib.sha256(NAMESPACE).hexdigest(),
        "cases": cases,
        "physical_reuse": physical_reuse,
        "qualified": not reasons and len(cases) == 2 and bool(physical_reuse),
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate prefix-cache namespace preservation across forks")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from vllm.core.block.prefix_caching_block import (
        PrefixCachingBlockAllocator,
    )

    report = build_report(
        lambda num_blocks: PrefixCachingBlockAllocator(
            num_blocks=num_blocks, block_size=BLOCK_SIZE))
    atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
