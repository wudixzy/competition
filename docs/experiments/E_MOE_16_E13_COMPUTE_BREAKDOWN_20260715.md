# E-MOE-16: Post-E-MOE-13 decode breakdown

## Purpose

E-MOE-12/13 reduced selected-weight gather substantially, so the older
E-MOE-07 decomposition no longer identifies the current bottleneck. E-MOE-16
remeasures the qualified single-token path with the production E-MOE-13
gather, fused `SiluAndMul`, and exact CoreX weighted reduction.

## Method

`tests/bench_moe_e13_breakdown.py` uses the checkpoint's TP4 rank-local shape
on GPU1. Each region ran 30 warmups and 9 repeats of 500 iterations. Inputs,
selected weights, and intermediate outputs were fixed so isolated operator
measurements do not include unrelated work.

## Results

| Region | Median ms | Share of routed full |
| --- | ---: | ---: |
| Route | 0.057203 | 17.31% |
| E-MOE-13 gather | 0.063454 | 19.20% |
| W13 linear | 0.130149 | 39.39% |
| Fused activation | 0.010454 | 3.16% |
| W2 BMM | 0.053722 | 16.26% |
| Exact reduce | 0.007288 | 2.21% |
| Compute without gather | 0.202231 | 61.20% |
| Fixed full | 0.265769 | 80.43% |
| Routed full | 0.330434 | 100.00% |

The isolated compute sum is 0.201613 ms, closely matching the measured
0.202231 ms compute chain. Inferred composition overhead is only 0.007546 ms,
so wrapping the same operators in another Python or C++ function cannot meet
the 5% boundary gate by itself.

The fixed and routed outputs were bit-exact (`max_abs=0`). Raw artifact:

```text
/root/competition/bench_runs/20260715_E_MOE_16/gpu1-breakdown.json
```

## Decision

The next high-value experiment must target W13 linear or jointly cover both
expert GEMMs. W13 alone is 39.4% of the current boundary; activation and reduce
are already too small for standalone work. E-MOE-07 rejected eight pointer
GEMVs, but did not test algorithm selection for the already gathered,
contiguous `2048x2048` W13 matrix.
