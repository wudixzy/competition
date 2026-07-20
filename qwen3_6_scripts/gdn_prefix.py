"""Shared GDN prefix-state cache contracts for the BI100 runtime."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


GdnPrefixKey = Tuple[int, bytes]
GdnCapturePoint = Tuple[int, GdnPrefixKey]

_VALID_POLICIES = {"fine32", "admission64", "off"}
GDN_KERNEL_CHUNK_TOKENS = 64
GDN_DIRECT_MIN_REPLAY_TOKENS = 2

_VALID_RESTORE_MODES = {"direct", "chunk64", "aligned"}


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise RuntimeError(f"invalid {name}={value!r}; expected one of: {allowed}")
    return value


def gdn_cache_policy_from_env() -> str:
    return _env_choice("BI100_GDN_CACHE_POLICY", "fine32", _VALID_POLICIES)


def gdn_restore_mode_from_env() -> str:
    return _env_choice(
        "BI100_GDN_RESTORE_MODE", "direct", _VALID_RESTORE_MODES)


def gdn_restore_alignment(restore_mode: str, block_size: int,
                          scheduler_chunk_tokens: int) -> int:
    """Return the content boundary required by a restore mode."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if restore_mode == "direct":
        return block_size
    if restore_mode == "chunk64":
        alignment = GDN_KERNEL_CHUNK_TOKENS
    elif restore_mode == "aligned":
        alignment = scheduler_chunk_tokens
    else:
        raise ValueError(f"unknown GDN restore mode: {restore_mode}")
    if alignment <= 0 or alignment % block_size != 0:
        raise ValueError(
            f"{restore_mode} GDN restore requires a positive alignment "
            f"divisible by block_size={block_size}; got {alignment}")
    return alignment


def make_prefix_key(block_count: int, digest: bytes) -> GdnPrefixKey:
    if block_count <= 0:
        raise ValueError("GDN prefix key requires at least one complete block")
    if not isinstance(digest, bytes) or len(digest) != 32:
        raise ValueError("GDN prefix digest must be exactly 32 bytes")
    return block_count, digest


def keys_from_block_hashes(block_hashes: Sequence[bytes]) -> List[GdnPrefixKey]:
    return [make_prefix_key(i + 1, digest)
            for i, digest in enumerate(block_hashes)]


def strict_prefix_block_count(token_count: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if token_count <= 1:
        return 0
    return (token_count - 1) // block_size


def key_at_strict_boundary(block_hashes: Sequence[bytes], token_count: int,
                           block_size: int) -> Optional[GdnPrefixKey]:
    block_count = min(
        len(block_hashes), strict_prefix_block_count(token_count, block_size))
    if block_count <= 0:
        return None
    return make_prefix_key(block_count, block_hashes[block_count - 1])


def final_capture_key(
        block_hashes: Sequence[bytes], prompt_tokens: int, block_size: int,
        restore_mode: str, replay_alignment: int) -> Optional[GdnPrefixKey]:
    if restore_mode == "direct":
        block_count = min(
            len(block_hashes), strict_prefix_block_count(
                prompt_tokens, block_size))
        if (block_count > 0
                and prompt_tokens - block_count * block_size
                < GDN_DIRECT_MIN_REPLAY_TOKENS):
            block_count -= 1
        if block_count <= 0:
            return None
        return make_prefix_key(block_count, block_hashes[block_count - 1])
    if restore_mode not in {"chunk64", "aligned"}:
        raise ValueError(f"unknown GDN restore mode: {restore_mode}")
    if (replay_alignment <= 0 or replay_alignment % block_size != 0
            or prompt_tokens <= 1):
        return None
    boundary_tokens = ((prompt_tokens - 1) // replay_alignment
                       * replay_alignment)
    block_count = min(len(block_hashes), boundary_tokens // block_size)
    if block_count <= 0:
        return None
    return make_prefix_key(block_count, block_hashes[block_count - 1])


def restore_key_is_eligible(
        key: GdnPrefixKey, prompt_tokens: int, block_size: int,
        restore_mode: str, replay_alignment: int) -> bool:
    """Return whether restoring ``key`` preserves the execution contract."""
    make_prefix_key(*key)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    boundary_tokens = key[0] * block_size
    remaining_tokens = prompt_tokens - boundary_tokens
    if remaining_tokens <= 0:
        return False
    if restore_mode == "direct":
        return remaining_tokens >= GDN_DIRECT_MIN_REPLAY_TOKENS
    if restore_mode not in {"chunk64", "aligned"}:
        raise ValueError(f"unknown GDN restore mode: {restore_mode}")
    return (replay_alignment > 0
            and boundary_tokens % replay_alignment == 0)


def capture_points_for_step(
        targets: Iterable[GdnPrefixKey], physical_context_tokens: int,
        logical_end_tokens: int, block_size: int) -> Tuple[GdnCapturePoint, ...]:
    if physical_context_tokens < 0 or logical_end_tokens < 0:
        raise ValueError("token positions must be non-negative")
    if logical_end_tokens <= physical_context_tokens:
        return ()
    selected = {}
    for key in targets:
        make_prefix_key(*key)
        boundary_tokens = key[0] * block_size
        if physical_context_tokens < boundary_tokens <= logical_end_tokens:
            selected[boundary_tokens - physical_context_tokens] = key
    points = tuple(sorted(selected.items()))
    if len(points) > 2:
        raise ValueError("at most two GDN capture points are allowed per step")
    return points


@dataclass(frozen=True)
class GdnCachePlan:
    restore_key: Optional[GdnPrefixKey] = None
    capture_points: Tuple[GdnCapturePoint, ...] = ()
    evict_keys: Tuple[GdnPrefixKey, ...] = ()


class GdnPrefixStatePolicy:
    """Scheduler-owned state index with deterministic worker actions."""

    def __init__(self, policy: str) -> None:
        if policy not in _VALID_POLICIES:
            raise ValueError(f"unknown GDN cache policy: {policy}")
        self.policy = policy
        self.capacity = {"fine32": 32, "admission64": 64, "off": 0}[policy]
        self._resident: OrderedDict[GdnPrefixKey, None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._resident)

    def resident_keys(self) -> Tuple[GdnPrefixKey, ...]:
        return tuple(self._resident)

    def contains(self, key: GdnPrefixKey) -> bool:
        return key in self._resident

    def select_restore(
            self, live_prefix_keys: Sequence[GdnPrefixKey],
            max_blocks: int) -> Optional[GdnPrefixKey]:
        if self.capacity == 0 or max_blocks <= 0:
            return None
        best = None
        for key in live_prefix_keys[:max_blocks]:
            if key in self._resident:
                best = key
        if best is not None:
            self._resident.move_to_end(best)
        return best

    def repeated_branch_candidate(
            self, live_prefix_keys: Sequence[GdnPrefixKey],
            max_blocks: int) -> Optional[GdnPrefixKey]:
        """Return a repeated raw-KV branch that lacks recurrent state.

        A live KV hit proves that the content occurred in an earlier request;
        the current request is therefore the second or later occurrence.
        """
        if self.policy != "admission64" or max_blocks <= 0:
            return None
        candidate = live_prefix_keys[min(len(live_prefix_keys), max_blocks) - 1]
        if candidate in self._resident:
            return None
        return candidate

    def admit(self, keys: Iterable[GdnPrefixKey]) -> Tuple[GdnPrefixKey, ...]:
        evicted: List[GdnPrefixKey] = []
        if self.capacity == 0:
            return ()
        for key in keys:
            make_prefix_key(*key)
            if key in self._resident:
                self._resident.move_to_end(key)
            else:
                self._resident[key] = None
            while len(self._resident) > self.capacity:
                evicted_key, _ = self._resident.popitem(last=False)
                evicted.append(evicted_key)
        return tuple(evicted)

    def forget(self, keys: Iterable[GdnPrefixKey]) -> None:
        for key in keys:
            self._resident.pop(key, None)
