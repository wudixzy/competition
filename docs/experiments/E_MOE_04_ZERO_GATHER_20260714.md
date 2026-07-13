# E-MOE-04 Zero-Gather Expert Views - 2026-07-14

## Hypothesis

The current single-token routed-MoE path copies the eight selected experts
before running one flattened W13 GEMM and one W2 batched GEMM. E-MOE-02 showed
that this gather accounts for roughly 43% of the isolated primitive. Directly
indexing each expert as a view could remove the copy at the cost of executing
eight W13 and eight W2 GEMMs.

The experiment changes only `tests/bench_moe_decode.py`. It does not change the
model or the fixed evaluator command.

## Method

Production TP-rank shape:

```text
experts                    256
hidden size                2048
intermediate per TP rank   128
top-k experts              8
dtype                      float16
```

All four BI100 devices ran five warmups followed by seven repeats of 50
iterations. `loop_views_fixed_ids` is an optimistic lower bound with expert IDs
already on the CPU. `loop_views_sync_ids` includes the required
`topk_ids.tolist()` device-to-host synchronization.

Artifacts:

```text
bench_runs/20260714_E_MOE_04/primitive/gpu0.json
bench_runs/20260714_E_MOE_04/primitive/gpu1.json
bench_runs/20260714_E_MOE_04/primitive/gpu2.json
bench_runs/20260714_E_MOE_04/primitive/gpu3.json
```

## Results

Median range across the four devices:

| Primitive | Median range | Speed relative to current |
| --- | ---: | ---: |
| Current advanced-index full path | 0.4371-0.4520 ms | 1.000x |
| Advanced-index gather only | 0.1846-0.1884 ms | 2.320-2.440x |
| Flat compute, preselected | 0.2423-0.2475 ms | 1.804-1.860x |
| Expert-view loop, fixed CPU IDs | 1.1959-1.2151 ms | 0.365-0.375x |
| Expert-view loop, synchronized IDs | 1.4588-1.5055 ms | 0.299-0.303x |

All four devices produced finite outputs. Current, double-bmm, and expert-loop
outputs had max absolute error 0.0 for the benchmark inputs. Routing IDs were
equal; the routing-weight max absolute difference was `2.98e-8`.

The current main service was also frozen before the primitive run with three
sequential 128-token groups. All requests succeeded. Output TPS P10 was
11.710, 11.936, and 11.861; TTFT P90 was 4.006, 3.871, and 3.833 seconds.

## Decision

**Reject.** Even the impossible fixed-ID lower bound is about 2.7 times slower
than the current path. The deployable synchronized-ID path is about 3.3 times
slower. Eliminating the memory copy does not compensate for replacing two
grouped operations with sixteen small GEMMs and adding a host synchronization.

No production code advances from this branch. Retain the flattened W13 plus W2
batched-GEMM implementation.
