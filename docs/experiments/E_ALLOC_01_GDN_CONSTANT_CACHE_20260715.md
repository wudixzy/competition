# E-ALLOC-01: Cache GDN chunk masks and identity

Date: 2026-07-15

## Hypothesis

Every GDN prefill call constructs two 64x64 boolean triangular masks and one
float32 identity matrix on the device. These tensors depend only on device,
chunk size, and dtype. Reusing immutable tensors should remove repeated
allocations and kernel launches without changing GDN math or model state.

## Manifest

```text
baseline: b1d95009d52135a5b00bbac1c5ccc682c4539644
branch: exp/E-ALLOC-01-gdn-constant-cache
model: /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
hardware: 4 x BI-V100-50C-200G, TP=4
production code changed: no
fixed evaluator contract changed: no
```

## Probe

`tests/bench_gdn_constant_cache.py` reproduces the production
`_torch_chunk_gated_delta_rule` at the real per-rank shape:

```text
batch=1, heads=12, key_dim=128, value_dim=128, chunk_size=64
input dtype=float16, recurrent state dtype=float32
```

It compares the current path, which constructs constants inside every call,
with a candidate that receives one prebuilt read-only tuple. T=64 exercises one
internal chunk and T=256 exercises four. The full rule includes Q/K
normalization, all decay/attention transforms, the triangular recurrence,
state update, and final output.

## Correctness

On all four devices and both token shapes:

- output maximum absolute difference: 0
- final-state maximum absolute difference: 0
- output equality: exact
- final-state equality: exact
- all outputs and states finite

## Results

All values are median milliseconds. Speedup is current divided by cached.

| GPU | Shape | Current full rule | Cached full rule | Speedup |
| --- | --- | ---: | ---: | ---: |
| 0 | T=64 | 6.0037 | 5.8940 | 1.019x |
| 1 | T=64 | 6.0141 | 6.0545 | 0.993x |
| 2 | T=64 | 5.9338 | 6.0582 | 0.979x |
| 3 | T=64 | 5.9285 | 6.0014 | 0.988x |
| 0 | T=256 | 7.5614 | 7.1355 | 1.060x |
| 1 | T=256 | 7.6525 | 7.5467 | 1.014x |
| 2 | T=256 | 7.6054 | 7.5098 | 1.013x |
| 3 | T=256 | 7.6234 | 7.5268 | 1.013x |

Building the constants costs 0.048-0.054 ms, while returning the cached tuple
costs about 0.00008 ms. That isolated saving is too small relative to the full
rule: median T=64 speedup is about 0.991x and median T=256 speedup is about
1.013x. T=64 regresses on three of four devices, and the T=256 result is below
the 5% primitive threshold on three devices.

Artifacts:

```text
bench_runs/20260715_E_ALLOC_01/gpu0.json
bench_runs/20260715_E_ALLOC_01/gpu1.json
bench_runs/20260715_E_ALLOC_01/gpu2.json
bench_runs/20260715_E_ALLOC_01/gpu3.json
```

## Decision

`REJECT` before production integration. The candidate is mathematically exact,
but the complete-operation benefit is below the experiment threshold and is
not directionally consistent at T=64. Do not pay a model restart, oracle, or
long-context qualification cost for this change. Keep the benchmark and
results as evidence, leave `integration/perf-winners` unchanged, and move to a
larger GDN prefill target such as the exact triangular recurrence.
