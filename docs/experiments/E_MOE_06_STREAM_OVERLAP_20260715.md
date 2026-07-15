# E-MOE-06: Routed/shared expert stream overlap

## Scope

The E-MOE-03 decode path computes the routed experts and shared expert
sequentially after the fused router/shared-gate projection. E-MOE-06 tests
whether placing either independent branch on an auxiliary CUDA stream can
hide shared-expert work without changing arithmetic, weights, routing, or the
final reduction order.

```text
base:  e4cf6cb (qualified E-MOE-03 model plus rejected experiment evidence)
bench: f326aa7
host:  ssh-a2d0a302.default.gpu.phanthy.com
```

## Primitive gate

`tests/bench_moe_stream_overlap.py` uses one decode token and the real per-rank
Qwen3.6-35B-A3B dimensions:

```text
hidden=2048, experts=256, top_k=8, local_intermediate=128, dtype=float16
router/shared gate=(257, 2048)
routed w13=(256, 256, 2048), routed w2=(256, 2048, 128)
shared w13=(256, 2048), shared w2=(2048, 128)
```

The benchmark compares the current sequential path with both possible stream
assignments. Each result is the median of nine trials, with 300 forwards per
trial after 20 warmups, on physical GPU1.

| Case | Median (ms) | P10 (ms) | P90 (ms) | Speedup | Exact | Max abs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| sequential | 0.783860 | 0.782842 | 0.784431 | 1.0000x | yes | 0.0 |
| shared branch on auxiliary stream | 1.087882 | 1.084319 | 1.127274 | 0.7205x | yes | 0.0 |
| routed branch on auxiliary stream | 1.101790 | 1.098989 | 1.113752 | 0.7114x | yes | 0.0 |

Both stream variants preserve bit-exact output, but they regress the complete
MoE block by 27.95% and 28.86%. On BI100, stream scheduling, synchronization,
and concurrent resource contention cost more than the overlap can hide.

## GPU state

The new instance connected successfully and exposed four BI-V100 cards. GPU1-3
were idle and GPU1 completed the benchmark with exit status 0. GPU0 again
reported 257 MiB allocated and 100% utilization despite `ixsmi` listing no
process. This repeats across instances and remains a platform/host-level issue,
but it does not affect the single-card GPU1 rejection result.

Remote artifacts are intentionally untracked:

```text
/root/competition/bench_runs/20260715_E_MOE_06/gpu1.json
/root/competition/bench_runs/20260715_E_MOE_06/gpu1.log
/root/competition/bench_runs/20260715_E_MOE_06/gpu1.status
```

## Decision

`REJECT AS PERFORMANCE WINNER`. The result is far below the 1.05x primitive
integration gate, so no production model patch, four-card collective test, or
service A/B is justified. Keep E-MOE-03 as the qualified model.
