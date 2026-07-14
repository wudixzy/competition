# E-GDN-03: Refined ixformer solve for the GDN triangular recurrence

Date: 2026-07-14

## Hypothesis

The 63 Python row updates in the GDN chunk rule compute the inverse of a unit
lower-triangular system. Replacing them with the vendor `ixformer.solve`
operation should reduce launch overhead while retaining the float32 state and
the fixed evaluator contract.

## Manifest

```text
baseline: b1d95009d52135a5b00bbac1c5ccc682c4539644
candidate: cb0861ba6b5ccc590c9ff97884277127cf45bf48
branch: exp/E-GDN-03-ixformer-solve
model: /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
hardware: 4 x BI-V100-50C-200G, TP=4
seed: 123
fixed evaluator contract changed: no
```

## Derivation and implementation

For the strictly lower-triangular matrix `A`, the original row recurrence
constructs `B` such that:

```text
B = A + A B
I + B = (I - A)^-1
```

`torch.linalg.solve_triangular` exists in the CoreX build but fails at runtime
because it looks for `/opt/sw_home/local/cuda/lib64/libcusolver.so`.
`ixformer.functions.solve` is available and accepts the real batched float32
shape when both the coefficient and right-hand-side tensors are contiguous.

One direct vendor solve accumulated enough float32 error for a T=256 chained
state comparison to diverge by about `3e-2`. The candidate therefore applies
one iterative-refinement step:

```text
X0 = solve(I - A, I)
R  = I - (I - A) X0
X  = X0 + solve(I - A, R)
```

A second refinement did not improve the observed error and was not retained.
The experiment also adds a CPU reference unit test, the existing GDN parity
fixture integration, static coverage, and a four-device primitive benchmark.

## Qualification gates

- per-device tensor preflight: 4/4 pass
- four-rank collective preflight: 4/4 pass, value `10.0`
- refined-solve unit tests: 3/3 pass
- GDN production/reference parity tests: 2/2 pass
- P0 static tests: 41/41 pass
- complete unit discovery: 126 pass, 1 skipped
- shell syntax, Python compilation, and `git diff --check`: pass
- installed runtime SHA equals repository SHA:
  `0647fe1379c434b48d498dd503a49a3bdfe4afd34a0941543a476243b81728f8`
- checkpoint loading: 26/26 shards
- candidate startup profile: pass, no non-finite value or exception
- full API smoke: 15/15 pass, including image, tool-call, JSON-schema, and
  prefix-cache cases
- candidate service error-log scan: zero fatal matches

The vendor solve does not mutate the cached identity RHS. A direct CoreX
probe reported finite output and zero measured post-refinement residual for a
batch of twelve 64x64 systems.

## Primitive results

The real per-rank primitive used 12 heads, key/value dimension 128, chunk size
64, float16 projections, float32 solve/state, and all four devices.

| Tokens | Current rule (ms) | Refined solve (ms) | Speedup | Output max abs | State max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 5.844-6.266 | 1.302-1.334 | 4.436-4.695x | 1.49e-8 | 2.38e-7 |
| 256 | 6.775-7.622 | 2.686-2.847 | 2.522-2.709x | 3.73e-8 | 3.58e-7 |

All primitive outputs and final states were finite. Artifacts:

```text
bench_runs/20260715_E_GDN_03/refined-gpu0.json
bench_runs/20260715_E_GDN_03/refined-gpu1.json
bench_runs/20260715_E_GDN_03/refined-gpu2.json
bench_runs/20260715_E_GDN_03/refined-gpu3.json
```

## Service correctness

The candidate started with 16,807 GPU blocks and 6,553 CPU blocks. The same
node and baseline reported 16,871 and 6,553, respectively. The retained
batched identity therefore costs 64 GPU blocks and reduces KV capacity.

Two of the three deterministic API oracle requests matched the baseline
exactly. The remaining greedy request was stable across three candidate
repeats but did not match the live baseline:

```text
baseline: 26 completion tokens
candidate: 16 completion tokens
finish reason: stop on both sides
prompt/cache accounting: equal
```

The live baseline reproduced the stored baseline text exactly. This is a real
floating-point decision divergence, not request randomness or stale cache.
It violates the project stop condition requiring output hashes to match.

## Long-prefill measurements

Every sample used four serial requests, prompt repeat 600, 16 generated
tokens, seed 123, and a cold first request followed by three prefix-cache hits.
Within each pair, success rate, prompt tokens, cached tokens, and completion
tokens match exactly.

| Sample | Weighted score change | TTFT P90 change | Wall change |
| --- | ---: | ---: | ---: |
| P1 candidate run 1 | -3.60% | +8.48% | +3.90% |
| P2 candidate run 1 | +7.44% | -8.59% | -7.20% |
| P3 candidate run 1 | +9.02% | -9.64% | -8.61% |
| P2 candidate run 2 | -4.02% | +12.56% | +4.38% |
| P3 candidate run 2 | +6.72% | -6.52% | -6.60% |

The five-sample weighted-score median is `+6.72%`, with three wins and two
losses. However, the same candidate's P2 score changed by `-10.67%` between
its two startups, and reverse-order performance was not consistently
positive. The endpoint result is too variable to qualify independently of the
correctness failure.

Artifacts:

```text
bench_runs/20260715_E_GDN_03/final-comparison.json
bench_runs/20260715_E_GDN_03/oracle-comparison.json
bench_runs/20260715_E_GDN_03/smoke-full.json
bench_runs/20260715_E_GDN_03/candidate-LONG*.json
bench_runs/20260715_E_GDN_03/baseline-LONG-P2.json
bench_runs/20260715_E_GDN_03/baseline-LONG-P3.json
```

## Decision

`REJECT` the production-path change. The primitive is mathematically
equivalent and substantially faster for T=64/T=256, but the candidate changes
a deterministic greedy output, retains enough RHS memory to lose 64 GPU KV
blocks, and does not provide stable endpoint gains across reverse-order
restarts. Do not merge it into `integration/perf-winners`.

The next exact experiment should preserve the original arithmetic order and
remove only redundant data movement, starting with the read-only
`attn[..., :i, :i].clone()` inside the recurrence.
