# E-MOE-13: Segmented 128-bit selected-weight gather

## Scope

E-MOE-12 made selected W13/W2 copying exact and profitable, but its half2
kernel still performed 64-bit division and remainder operations for each
copied element. E-MOE-13 preserves the same runtime API and output tensors
while changing only the copy schedule:

- one two-dimensional launch maps `grid.y` to 8 W13 and 8 W2 expert slices;
- each thread copies one aligned 16-byte `uint4` value;
- expert and output offsets are computed once outside the copy loop;
- the measured production grid is fixed at `grid=(8, 16)` with 256 threads.

Routing, selected expert order, GEMMs, activation, reduction, prefill, and the
fixed evaluator command remain unchanged. The existing
`BI100_MOE_COREX_WEIGHT_GATHER=0` fallback still restores native indexing.

## Method

`tests/bench_moe_weight_gather_vec.py` compares E-MOE-13 directly against the
qualified E-MOE-12 production extension at the checkpoint's TP4 rank-local
shape:

```text
experts=256, top_k=8, hidden=2048, local_intermediate=128, dtype=float16
```

GPU1 first scanned `grid_x={8,16,32,64,128}`. Each measurement used 30
warmups and 9 repeats of 300 iterations. The winning `grid_x=8` was then run
on GPU1-3 with 1,000 random routed exactness steps per device. The existing
production-method probe separately compiled and executed the final production
source and runtime dispatch.

## Results

| GPU | E-MOE-12 gather ms | E-MOE-13 gather ms | Gather speedup | E-MOE-12 routed ms | E-MOE-13 routed ms | Routed speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.093954 | 0.065274 | 1.439x | 0.363647 | 0.333530 | 1.090x |
| 2 | 0.094009 | 0.065474 | 1.436x | 0.367404 | 0.335816 | 1.094x |
| 3 | 0.092870 | 0.064118 | 1.448x | 0.360799 | 0.329471 | 1.095x |

The fixed-route complete-path speedups were `1.114x`, `1.125x`, and `1.126x`.
Every W13 tensor, W2 tensor, and final output was bit-exact. All three devices
passed 1,000/1,000 random routed comparisons with `max_abs=0`.

The separately compiled production source and production model-method probe
on GPU1 reported:

```text
native advanced-index method: 0.471223 ms
E-MOE-13 production method:   0.337556 ms
speedup:                      1.3960x
fixed exact:                  true, max_abs=0
sequence exact:               1000/1000, max_abs=0
```

The cross-device median incremental saving over E-MOE-12 is 0.03133 ms per
MoE layer, or about 1.25 ms/token over 40 layers. Combined with E-MOE-12, the
selected-weight optimization contributes approximately 5.01 ms/token beyond
E-MOE-11. The conservative short-context candidate-stack projection therefore
increases from 8.9 to about 10.1 ms/token, implying roughly 15.4-15.6 TPS from
the current 13.3-13.5 TPS baseline. TP4 service A/B remains authoritative.

Raw artifacts are intentionally not committed:

```text
/root/competition/bench_runs/20260715_E_MOE_13/cross-gpu1.json
/root/competition/bench_runs/20260715_E_MOE_13/cross-gpu2.json
/root/competition/bench_runs/20260715_E_MOE_13/cross-gpu3.json
/root/competition/bench_runs/20260715_E_MOE_13/production-dispatch-gpu1.json
```

## Decision

`QUALIFY FOR TP4 SERVICE A/B AND SUPERSEDE THE E-MOE-12 COPY KERNEL`. The
incremental three-device routed gain is stable at 9.0%-9.5%, exceeds the 5%
gate, and retains the exact-output contract. The feature switch and fallback
semantics are unchanged.

Physical GPU0 on `ssh-a2d0a302.default.gpu.phanthy.com` still times out on a
small CUDA tensor operation. GPU1-3 are healthy, but TP4 service qualification
requires a healthy four-card instance or a host-side reset.
