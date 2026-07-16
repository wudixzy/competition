# M1-21 Prefix-Cache Trace Design

## Status

`DIAGNOSTIC ONLY`. This branch must not replace the qualified submission until
a real evaluator trace proves a retention-policy gain. The current production
`main` remains unchanged.

## Why a trace is required

The published workload describes 881 streaming requests, about 768 session
threads, a median of three requests per thread, and a theoretical token-weighted
reusable prefix near 65.6%. It does not publish the ordered block identities.
The prior official run reported only 42% cache hit, while M1-20's deliberately
balanced cold/warm matrix reported 49.93% by construction.

Neither result can predict an alternative eviction policy. Cache retention is
determined by the exact interleaving of sessions, prefix identities, request
lengths, and the fixed GPU block capacity. Changing LRU from aggregate
statistics would be blind tuning and could reduce both hit rate and correctness.

## Current vLLM behavior

The CoreX vLLM 0.6.3 runtime uses `BlockSpaceManagerV2` and
`PrefixCachingBlockAllocator`. Each full 16-token block receives a chained
content hash over itself and all previous blocks. On request completion, unused
computed blocks enter `LRUEvictor` with their last-access timestamp. Eviction
chooses the oldest timestamp; among blocks with the same timestamp it evicts
the deepest prefix block first.

The tie-break already protects shallow blocks within one completed request.
It does not protect an older, frequently reused shallow prefix from a more
recent long unique request. A frequency-aware policy is therefore plausible,
but only an ordered trace can quantify it.

## Trace contract

When and only when `BI100_CACHE_TRACE=1`, initial block-table allocation emits
one compact line per request:

```text
[BI100_CACHE_TRACE] {"version":1,...}
```

The payload contains:

- SHA-256 prefix of the request id, never the raw id;
- prompt length, block size, and GPU capacity;
- count of full immutable blocks;
- little-endian base64 encoding of the ordered 64-bit chained content hashes.

It does not log token ids, messages, tools, images, or model outputs. Hashes are
already computed by the allocator, so tracing performs no CUDA work and does
not change allocation, hit detection, or cached-token accounting. The branch is
still diagnostic because JSON/base64 formatting and log I/O can affect TTFT.

## Offline gate

The analyzer replays requests with concurrency one and compares:

1. current vLLM LRU, including the deeper-prefix tie-break;
2. one fixed frequency-aware policy: lowest observed frequency, then oldest,
   then deepest.

Both policies count only the longest contiguous cached prefix at request
arrival. Current-request blocks remain protected during admission. The
simulator reports block/token hit rates and does not invent Cache TPS without a
caller-supplied baseline.

Implementation of a production policy is authorized only if the complete 881
trace shows all of the following:

1. predicted token hit rate above 50%;
2. at least five percentage points over current LRU;
3. at least 5% weighted-score improvement using measured Cache TPS;
4. no request longer than available capacity and no reliance on hash gaps;
5. one fixed policy, with no per-layer/session/length threshold scan.

Otherwise close M1-21 and retain the current block manager.

## Implementation validation

The diagnostic branch now contains an idempotent fail-fast patch and a
standard-library offline simulator. Eight focused tests cover patching twice,
unknown vendor layouts, patched-source compilation, trace privacy/encoding,
longest-prefix gaps, capacity overflow, frequency-aware churn, and weighted
projection. The full static gate plus focused tests passed 60/60. Broad unit
discovery passed every available test and stopped only at the known optional
Pillow import (`test_clients_unit`).

The patch was also applied twice to an untouched CoreX vLLM 0.6.3
`block_manager_v2.py` copy on the BI100 host. The first run patched all three
anchors, the second reported all three as already patched, and `py_compile`
passed. Evidence is under:

```text
/root/M1_21_cache_trace_validation/
```

`Dockerfile` enables `BI100_CACHE_TRACE=1` only on this diagnostic branch.
Do not use its latency or score as a production comparison. After collecting
the complete evaluator log, run:

```bash
python3 scripts/analyze_prefix_cache_trace.py docker.log \
  --baseline-cache-tps 7716.5 \
  --out m1_21_cache_simulation.json
```

The baseline Cache TPS is the M1-20 local aggregate and remains only a local
weighted upper-bound input. Official Cache TPS should replace it when available.
