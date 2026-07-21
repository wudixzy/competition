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


SCHEMA = "bi100-prefix-namespace-fork-gate-v1"
VERSION = 1
BLOCK_SIZE = 4
BLOCK_TOKEN_IDS = ([1, 2, 3, 4], [5, 6, 7, 8])
NAMESPACE = b"bi100-prefix-fork-gate"


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


def evaluate_case(allocator_factory: Callable[[], Any],
                  release_order: str) -> dict[str, Any]:
    allocator = allocator_factory()
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


def build_report(allocator_factory: Callable[[], Any]) -> dict[str, Any]:
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

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "namespace_sha256": hashlib.sha256(NAMESPACE).hexdigest(),
        "cases": cases,
        "qualified": not reasons and len(cases) == 2,
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
        lambda: PrefixCachingBlockAllocator(num_blocks=8,
                                             block_size=BLOCK_SIZE))
    atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
