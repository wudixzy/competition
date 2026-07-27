# M1-67: Exact GDN q/k mapping plus value cast

## Hypothesis

The strict M1-65 decode path maps normalized FP16 q/k heads into expanded FP32
heads, then launches a separate FP16-to-FP32 cast for the value heads. One
production-shape kernel can perform those independent conversions in a single
allocation and launch without changing arithmetic.

This is a bounded component experiment. It does not change the model path,
prebuilt submission extension, runtime defaults, `computility-run.yaml`, or
request semantics.

## Fixed boundary

The only candidate operation is:

```text
reference: qk_map(normalized_q, normalized_k) + value.float()
candidate: qkv_map(normalized_q, normalized_k, value)
```

The shape is fixed to one TP4 rank of Qwen3.6-35B-A3B decode:

- batch: 1
- key heads: 4
- value heads: 8
- head dimension: 128
- input dtype: FP16
- output dtype: FP32

The q/k instructions are the existing mapping instructions. The additional
value conversion is `__half2float`, which is exactly representable in FP32.
There are no tile, block, threshold, dtype, or YAML scans.

## Component gate

`tests/bench_gdn_exact_qkv_map.py` compares two fixed seeds and 1,000
multi-scale random steps. Every complete q/k/v output must be finite and
bit-exact. Relative L2 must also be at most `1e-5`.

The alternating paired benchmark requires both:

- median and paired-median speedup at least `1.25x`;
- absolute saving at least `0.02 ms` per GDN layer.

Failure of either performance limit closes this direction without production
integration. A pass only permits a later production-boundary experiment; it
does not authorize a default switch or promotion.

## Lifecycle

`scripts/run_gdn_exact_qkv_map_gate.sh` requires a clean source tree and an
immutable runtime overlay. Build and benchmark commands run in process groups
created by the invocation. Cleanup sends `SIGTERM`, waits 60 seconds, sends
`SIGKILL` only to surviving members, and reaps the leader.

The exit trap then checks residual API/worker and GPU processes, scans
fatal/OOM/Gloo/NCCL/worker-loss and timeout signatures, repeats the selected
GPU preflight, and compares pre/post GPU state. Any cleanup or postflight
failure invalidates the component result.

## Current status

`HARNESS READY; REMOTE RESULT PENDING`.

The current TP4 host has only physical GPU1 passing the basic CUDA preflight.
M1-67 may use GPU1 for this isolated component gate, but no model or TP4 result
may be inferred from it.
