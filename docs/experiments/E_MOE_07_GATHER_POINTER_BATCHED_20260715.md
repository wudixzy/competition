# E-MOE-07: Selected-weight gather and pointer-batched GEMV

## Scope

The qualified T=1 routed-expert path selects eight of 256 experts, gathers
their W13 and W2 tensors, executes one flattened W13 linear and one W2 BMM,
then performs the weighted reduction. E-MOE-07 first decomposes that path and
then tests whether cuBLAS pointer-array GEMV can read the selected experts
directly from the original weight tensors.

The real per-rank decode dimensions are:

```text
experts=256, top_k=8, hidden=2048, intermediate=128
w13=(256, 256, 2048), w2=(256, 2048, 128)
selected weight traffic=12 MiB per layer and token
```

All tests ran serially on physical GPU1 of
`ssh-a2d0a302.default.gpu.phanthy.com`. GPU0 remained excluded because its
unowned 100% utilization condition was not resolved.

## PyTorch path decomposition

`tests/bench_moe_gather_breakdown.py` uses the qualified top-k-logits routing
and reports nine repeats of 300 iterations:

| Region | Median (ms) | P10 (ms) | P90 (ms) | Share of full |
| --- | ---: | ---: | ---: | ---: |
| Route only | 0.057159 | 0.057137 | 0.057217 | 11.27% |
| Gather only | 0.181279 | 0.181068 | 0.181373 | 35.75% |
| Compute with pregathered weights | 0.240700 | 0.240647 | 0.240948 | 47.47% |
| Gather plus compute | 0.430154 | 0.429800 | 0.430281 | 84.84% |
| Full current path | 0.507033 | 0.505600 | 0.508405 | 100.00% |

All three output comparisons are bit-exact. The direct gather measurement is
35.75% of the full path. The residual
`full - route - pregathered_compute` is 0.209174 ms, or 41.25%, and includes
gather plus composition/launch effects. This passed the gate for building a
no-copy prototype.

## cuBLAS no-copy prototype

`tests/corex_moe_pointer_batched_ext.cu` builds device pointer arrays from the
eight expert IDs and calls cuBLAS directly against the original weight
tensors. `tests/bench_moe_pointer_batched.py` preserves PyTorch routing, SiLU,
and weighted reduction so the experiment isolates the two selected-expert
matrix operations. Both native half accumulation and FP32 accumulation were
tested.

| Case | Median (ms) | P10 (ms) | P90 (ms) | Speedup | Exact | Max abs |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Current gather path | 0.505467 | 0.505209 | 0.505631 | 1.000x | yes | 0 |
| Pointer `HgemmBatched` | 0.805183 | 0.804415 | 0.805510 | 0.628x | no | 1.52588e-4 |
| Pointer GEMM, FP32 accumulate | 0.718231 | 0.717819 | 0.718673 | 0.704x | yes | 0 |

The FP32 accumulation case is bit-exact but 42.1% slower than the current
path. The half case is slower still and changes output values. Avoiding the
12 MiB gather does not compensate for two small pointer-batched GEMV calls on
this runtime; the flattened W13 linear and gathered W2 BMM remain materially
more efficient.

Remote evidence is intentionally untracked:

```text
/root/competition/bench_runs/20260715_E_MOE_07/breakdown.json
/root/competition/bench_runs/20260715_E_MOE_07/breakdown.log
/root/competition/bench_runs/20260715_E_MOE_08/build/result.json
/root/competition/bench_runs/20260715_E_MOE_08/run.log
```

## Decision

`REJECT AS PERFORMANCE WINNER`. Keep the decomposition and extension probe as
negative evidence, but do not integrate the pointer-batched path into the
model and do not spend a TP4/service qualification cycle on it. A successor
must use a genuinely fused BI kernel that combines selected-weight reads,
both expert projections, activation, and reduction; wrapping the same GEMVs
in more Python or cuBLAS calls is not a viable direction.
