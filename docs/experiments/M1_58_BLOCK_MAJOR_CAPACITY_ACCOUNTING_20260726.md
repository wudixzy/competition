# M1-58 Block-Major Capacity Accounting

## Conclusion

M1-57's block-major CacheEngine integration allocated its fixed GPU staging
after vLLM had already profiled memory and selected the GPU KV block count.
That left `167,772,160` bytes per rank outside the
`gpu-memory-utilization=0.9` capacity calculation. A healthy component smoke
could therefore pass while a full TP4 startup remained exposed to an avoidable
OOM or allocator-fragmentation failure.

M1-58 now deducts exactly 1,024 production KV blocks before the worker reports
available capacity:

```text
2 staging buffers
* 512 blocks
* 10 attention layers
* 2 KV planes
* 4096 FP16 elements
= 167,772,160 bytes
= 1,024 blocks at 163,840 bytes per block
```

The selector still defaults off. With
`BI100_BLOCK_MAJOR_CPU_KV=0`, the profiled block count is returned unchanged.
With the selector enabled, startup also requires
`BI100_CPU_KV_OFFLOAD=1`,
`BI100_HYBRID_KV_ACCOUNTING=full_attention`, and an exact 163,840-byte cache
block. Invalid dependencies, geometry, or a reservation that leaves no usable
blocks fail startup.

## Runtime Identity

The immutable bare-host runtime report and verifier now bind:

- `vllm/block_major_kv_cache.py`;
- `vllm/corex_block_major_kv_transfer.so`;
- the generated `vllm/worker/cache_engine.py`;
- the generated `vllm/worker/worker.py`;
- explicit CacheEngine and worker-capacity patch markers.

The fixed service contract records the CPU offload, block-major, and
block-major trace selectors. The startup gate can require either no
block-major reports for a control or exactly one capacity and cache allocation
report per TP rank for a candidate. Rank-local capacity may differ, but its
minimum must equal the final engine GPU block count.

## Validation

- focused capacity, patch, startup-contract, and runtime-identity tests:
  26 passed;
- complete local unit suite: 707 passed, 25 skipped;
- submission preflight: 9/9;
- formal command: 29 arguments, unchanged;
- formal environment: 3 entries, unchanged;
- official YAML and default selectors: unchanged.

This is an offline correctness and startup-safety qualification. The current
host has only three healthy cards and cannot run this model at TP3, so no model
startup, 262144 capacity, output correctness, quality, end-to-end latency, or
official score is claimed. Those remain mandatory TP4 gates.
