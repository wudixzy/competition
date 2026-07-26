# M1-57 Block-Major CacheEngine Integration

## Conclusion

The non-default block-major CPU KV path passed the same fixed real CoreX
`CacheEngine` integration gate on physical GPU1, GPU2, and GPU3. On every card,
a 1,025-block transfer crossed three 512-block chunks through the patched
`CacheEngine.swap_out/swap_in` methods and restored all ten layer tensors
byte-for-byte. Same-GPU-slot victim preservation and replacement, invalid-map
zero-write behavior, strict selectors, pinned memory, and compatibility views
also passed on all three cards.

This qualifies the integration for a healthy TP4 service gate. It does not
qualify model capability, 262144 capacity, end-to-end performance, the official
881 workload, `main`, or `computility-run.yaml`.

## Implementation

Branch `exp/M1-57-block-major-cache-engine-20260726` contains:

- `4263344`: production module, vendor patch, pinned binary, installer, and
  unit/preflight coverage;
- `6ad5ffe`: fixed single-GPU integration harness.

`BI100_BLOCK_MAJOR_CPU_KV=0|1` defaults to `0` and accepts no alternate
spellings. When enabled, startup requires exactly ten contiguous FP16 CUDA
caches with shape `[2, blocks, 4096]`, a positive CPU block count, and pinned
host memory. Unsupported geometry or a missing extension fails startup.

The canonical CPU pool has shape `[cpu_blocks, 10, 2, 4096]`. Ten non-owning
views preserve the public `[2, cpu_blocks, 4096]` layer shapes. The configured
CPU block count and bytes per block remain unchanged; there is no second CPU
cache allocation.

Every map is checked on CPU before any write:

- CPU tensor, contiguous `int64`, shape `[N, 2]`;
- source and destination in range;
- unique source and destination IDs.

The extension repeats bounds checks on GPU. D2H completes before H2D at the
worker level, and the two fixed staging events guard every slot reuse. The final
extension error check synchronizes the stream before the next CacheEngine call.
Optional `BI100_BLOCK_MAJOR_CPU_KV_TRACE=1` records only direction, block
count, byte count, and elapsed time.

## Runtime Identity

The remote could not authenticate to private GitHub, and its SCP subsystem
closed larger transfers. The gate therefore used a minimal isolated overlay,
not the complete model runtime:

- source revision: `6ad5ffe`;
- source archive SHA-256:
  `70808d44b62fe9747fa051ca6c8b89c7db710e12e090662240cfa29892f8bf24`;
- module SHA-256:
  `e8b63cdcf83b4e5b388519ddea3e7508770601d0761a6eb8b088e896bb3da99f`;
- harness SHA-256:
  `968eb7ba1d462cb4b27966f6a2fa2f85704e6493a250d276b0ea36e756f8786b`;
- extension SHA-256:
  `7e2aafd8dc755b0ee16c3b9bb812b95548fc042bbaa840dd9db7d2c51a10474c`;
- system CacheEngine SHA-256:
  `f69f6867a2321d59e2142012f17a04a49faa56b699535221e1c9fdb9d706e4bd`;
- patched overlay CacheEngine SHA-256:
  `2b9c72dd954476dc5446d84a3318e6defc447f56b13c1bc60a7a02afe80ea331`.

The system package was not modified. The real vendor CacheEngine patch reported
three successful replacements on the first run and three idempotent skips on
the second.

## Gate Result

| Check | GPU1 | GPU2 | GPU3 |
|---|---:|---:|---:|
| GPU / CPU blocks | 1,025 / 1,536 | 1,025 / 1,536 | 1,025 / 1,536 |
| CPU pool | `[1536,10,2,4096]` | `[1536,10,2,4096]` | `[1536,10,2,4096]` |
| Allocation | 262.476 ms | 263.765 ms | 266.440 ms |
| D2H + H2D round trip | 72.525 ms | 73.943 ms | 76.314 ms |
| Round-trip byte exact | pass | pass | pass |
| Same-slot victim/request | pass / pass | pass / pass | pass / pass |
| Invalid mapping fail-fast / zero-write | pass / pass | pass / pass | pass / pass |
| Default off / invalid selector fail-fast | pass / pass | pass / pass | pass / pass |
| Fatal, OOM, Gloo, worker loss, traceback | none | none | none |

Every card reported `335,708,672` allocated GPU bytes, including the
1,025-block synthetic GPU KV fixture and both staging buffers. The production
candidate's incremental GPU allocation is the two fixed staging buffers,
`167,772,160` bytes per rank. TP4 startup must measure its actual effect on
reported GPU block capacity.

These round-trip measurements are correctness smokes, not a performance
comparison. M1-56 remains the performance evidence for the data plane.

Raw evidence and SHA-256:

- GPU1: `docs/experiments/evidence/M1_57_CACHE_ENGINE_INTEGRATION_GPU1_20260726.json`,
  `c66dd831a0b96f1b5691a26cd8f301498d5cdc89060941c33732150204a87a95`;
- GPU2: `docs/experiments/evidence/M1_57_CACHE_ENGINE_INTEGRATION_GPU2_20260726.json`,
  `7672b1d532ab3c6daf8d7fd97c80d94536764b129cfdd940a76de6f32b01a005`;
- GPU3: `docs/experiments/evidence/M1_57_CACHE_ENGINE_INTEGRATION_GPU3_20260726.json`,
  `0239a37aa4c0d355a9fe2726f7a97114e84ea58be3685bcbaf5d45366858a383`.

## Remaining TP4 Gates

The next A/B must use one full hash-pinned runtime and four healthy cards.
Control and candidate may differ only in the block-major selector after fixing
the already-qualified hybrid accounting and content-addressed CPU tier
settings.

Required results remain:

1. startup identity, four-rank agreement, staging allocation, and actual GPU/CPU
   block counts;
2. fixed eviction-pressure D2H/H2D activity with same-slot correctness;
3. cold/warm greedy token identity and effective KV/GDN hit intersection;
4. 65K, 131K, 235K, and 262144 capacity;
5. complete functional, tool, reasoning, multimodal, and quality gates;
6. Output TPS, Input TPS, Cache TPS, TTFT, hit rate, success rate, and weighted
   score from an admissible workload.

Until all applicable gates pass, the selector remains absent from the official
YAML and no default or `main` change is justified.

## Local Validation

- full unit suite: 702 passed, 25 skipped;
- submission preflight: 9/9;
- prebuilt extensions: 11 hash-pinned ELF files;
- formal command and environment: unchanged.
