# M1-41 Content-Frequency KV Evictor

## Purpose

M1-41 requalifies the fixed M1-29 frequency-aware eviction architecture after
the cache correctness work changed prefix identity from process-local integer
hashes to chained SHA-256 content keys. It is the remaining cache architecture
candidate; it is not a capacity, decay, threshold, or YAML scan.

The fixed victim key is:

```text
(content frequency ascending,
 last access ascending,
 logical prefix depth descending,
 physical block id ascending)
```

Frequency belongs to the 32-byte logical content key and survives physical KV
block reuse. A generation-validated lazy heap keeps eviction O(log N), and a
fixed `2 * live_blocks + 1` compaction bound prevents stale heap growth.

## Runtime Isolation

`BI100_KV_EVICTION_POLICY=lru|frequency` is an internal private-branch switch.
The default is `lru`; invalid values fail during allocator construction. The
experimental evictor is installed by the Docker patch path so the branch stays
buildable, but the selector is explicitly forbidden in `computility-run.yaml`
until qualification passes.

The runtime does not alter token hashes, GDN state identity, numerical kernels,
KV capacity, or evaluator arguments. Effective cached tokens remain the
intersection of contiguous resident KV and a restorable GDN state.

## Predeclared Gates

Implementation gates:

- 10,000 deterministic mixed lifecycle operations match a full-scan oracle;
- real `CpuGpuBlockAllocator` cache contents, frequency counts, and hit counts
  match a full-scan oracle after each of 600 requests;
- every runtime frequency key is exactly 32 bytes;
- three fixed 881-request proxy runs have identical hits, evictions, and final
  cache digest;
- heap entries never exceed `2 * live_blocks + 1` in runtime tests and
  `2 * capacity + 1024` in the fixed process-level proxy;
- preserve the M1-29 CPU gates: mean added time versus its frozen LRU run no
  greater than `3 ms/request`, P90 no greater than `10 ms/request`;
- isolated process RSS no greater than `256 MiB`;
- default LRU behavior and API remain unchanged.

Qualification gates require one complete trace from a single runtime session:

- v4 SHA trace ordinals exactly `1..881` with matching same-run metrics;
- `admission64 + frequency` effective KV/GDN hit rate gains at least five
  percentage points over `fine32 + LRU`;
- per-request residual-prefill projection improves weighted score by at least
  5%, without aggregate hit-rate scaling;
- Output TPS P10 remains at least 20 and regresses no more than 2%;
- success at least 99%, effective hit at least 50%, projected TTFT P90 at most
  5 seconds, projected weighted score at least 8000;
- TP4 numerical replay and 262144 capacity pass before any YAML or main change.

Synthetic proxy hits cannot satisfy qualification. If a complete trace does
not pass the hit and score gates, stop this direction without trying frequency
decay, admission thresholds, alternative tie breaks, or cache-size changes.

## Status

`IMPLEMENTATION_GATES_PASSED; TRACE_QUALIFICATION_PENDING`.

The fixed 32-byte content-key proxy completed three runs with identical
`27,483` hits, `1,392,410` evictions, and final cache SHA-256. Median request
time was `3.500 ms`; P90 was `10.615 ms`. The isolated process used
`169.125 MiB` peak RSS, and the heap peaked at `21,056` entries versus the
fixed `34,780` bound. All CPU, determinism, heap, and memory checks passed.

The real installed CoreX vLLM allocator differential then passed all 600
requests:

| Seed | Capacity | Block size | Requests | Hit blocks | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 20260721 | 12 | 4 | 200 | 185 | pass |
| 7 | 7 | 4 | 200 | 149 | pass |
| 99 | 16 | 8 | 200 | 124 | pass |

After every request, the candidate and full-scan oracle had identical
contiguous hits, cached SHA content map, global frequencies, and bounded heap.
The imported runtime was
`/usr/local/corex/lib64/python3/dist-packages/vllm/__init__.py`.

Evidence:

- `docs/experiments/evidence/M1_41_CONTENT_FREQUENCY_PROXY.json`
  (`0306638339a9d9153d6a24583bcf5d179a9d9ac9d30ddb5bc326303ff719895d`)
- `docs/experiments/evidence/M1_41_ALLOCATOR_DIFFERENTIAL.json`
  (`468fb8433fe5411dff805dd37f022cc4ec0426ab5b419926133bfc86a257c9bb`)

These results authorize retaining the default-off candidate, not enabling it.
Production `main`, submission YAML, and the running service remain unchanged.
The missing decision input is a complete single-session v4 trace with ordinals
`1..881` and same-run metrics. The current instance is also ineligible for TP4
because GPU3 still times out.

A bounded inventory checked the remote `bench_runs`, prior M1-32 directories,
and local `result` tree. It found no v4 trace records and no single-session
ordinal sequence; the only marker matches were source and design documents.
Generating the missing trace requires the evaluator workload or an equivalent
881-request replay and cannot be reconstructed from published bucket
aggregates. M1-41 therefore remains default-off rather than inferring a score
from its synthetic proxy.
