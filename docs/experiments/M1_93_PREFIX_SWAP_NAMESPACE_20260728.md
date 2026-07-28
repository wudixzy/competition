# M1-93 request-swap prefix namespace

Date: 2026-07-28

Branch: `fix/M1-93-prefix-swap-namespace-20260728`

## Problem

The content-addressed prefix cache binds the first full block to a runtime,
adapter, and multimodal namespace. Request-level swap-out preserves the
logical block object, but the old swap-in path rebuilt its temporary first
block through `allocate_immutable_block()`, which hard-coded an empty
namespace.

That created two conflicting identities:

- the logical block retained its original namespaced content hash;
- the destination allocator registered the physical block under the empty-
  namespace hash.

A later release, swap, or reuse could then miss the allocator entry or fail an
internal assertion. The same path also rebuilt partial blocks without
explicitly carrying their namespace.

The fixed competition command uses `max_num_seqs=1` and normally recomputes
instead of request-level swap, so this is not claimed as the cause of the
reported platform score. It is a correctness defect in a supported runtime
path, especially for multimodal or adapter requests.

## Fix

`PrefixCachingBlockAllocator.swap_in()` now rebuilds:

- full blocks with
  `allocate_immutable_block_with_cache_namespace()`; and
- partial blocks with
  `allocate_mutable_block_with_cache_namespace()`.

Both calls receive the logical block's existing `cache_namespace`. Token IDs,
content hashing, physical allocation, transfer mappings, request semantics,
cache policy, model execution, and formal configuration are unchanged.

## Gates

The source-level behavioral test extracts the production `swap_in()` method
and requires both full and partial blocks to pass their original namespace to
the namespace-aware allocator methods. It also checks token append behavior,
physical ID transfer, and temporary-block release.

The installed-overlay cache namespace gate is upgraded to v3. Its tenth check
performs a two-block namespaced round trip across independent source and
destination allocators:

1. allocate under one non-empty namespace;
2. swap source to destination;
3. require the destination allocator to register both original content
   hashes under the transferred physical IDs;
4. swap back and require the same invariant at the source;
5. require the logical hash chain and namespace to remain unchanged.

The gate additionally binds the installed prefix allocator module SHA-256.
M1-87 is upgraded to v5 and rejects an old v2 cache report, a missing check,
or a missing allocator digest.

## Admission transaction audit

The same audit noted that the scheduler updates its GDN resident index when it
constructs capture actions, before workers finish model forward and save the
state. In the current synchronous engine, an exception from
`model_executor.execute_model()` propagates out of the engine step; the
service cannot continue scheduling requests with that in-memory mismatch, and
a restart resets both indexes. Adding a scheduler commit protocol without a
real worker acknowledgement would therefore add complexity without closing a
demonstrated continuing-service failure.

This remains a documented fail-stop boundary. Any future request-level model
error recovery must first add an explicit all-rank capture acknowledgement,
then commit scheduler admission and eviction. It must not silently continue
from an unconfirmed state.

## Evidence boundary

Local validation completed:

- focused namespace, M1-87, GDN, and static tests: 134 passed with one
  optional Pillow test skipped;
- complete tests-root discovery: 1080 passed with 25 dependency-gated skips;
- submission preflight: 9/9 passed;
- quality-data manifests: 12 long-context and 11 Agent cases passed;
- official metric manifest: 53 cases passed;
- Python/shell syntax, line-ending, sensitive-artifact, and diff checks
  passed.

These tests exercise source behavior and evidence contracts. The installed v3
round-trip must still run against an immutable CoreX overlay.
No model, GPU, TP4, semantic-quality, long-context, throughput, official
881-request, or score result is claimed.

The branch does not change `computility-run.yaml`, model defaults, `main`, or
repository visibility. A healthy BI100 should run M1-91 first and then the
M1-87 v5 single-GPU queue before any TP4 promotion work.
