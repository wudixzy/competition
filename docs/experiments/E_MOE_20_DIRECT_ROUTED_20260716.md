# E-MOE-20 direct routed-expert prototype

## Scope

E-MOE-20 replaces the current T=1 selected-weight gather, W13 GEMM, W2 BMM,
and routed reduction with shape-specific direct-addressing kernels. It does
not change routing and does not alter prefill. This first result is a
single-GPU algorithm gate, not a production or TP4 result.

Tested rank-local shape:

```text
experts=256, top_k=8, hidden=2048, intermediate=128, dtype=FP16
```

The staged candidate uses direct W13, the existing `SiluAndMul`, then direct
W2 plus routed-weight reduction. The fused candidate also folds activation
into the W13 stage.

## GPU1 result

Physical GPU1 on `ssh-9cd6a034` passed compilation, short smoke, nine timing
repeats, and a 500-step random numerical sequence. Only that physical GPU was
visible to the process.

| Boundary | Baseline ms | Staged ms | Staged speedup | Fused ms | Fused speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed expert path | 0.263837 | 0.043937 | 6.005x | 0.037477 | 7.040x |
| Routing plus expert path | 0.331490 | 0.112576 | 2.945x | 0.102432 | 3.236x |

Component medians explain the gain:

| Current component | ms | Direct replacement | ms |
| --- | ---: | --- | ---: |
| selected W13/W2 gather | 0.070440 | removed | 0 |
| W13 | 0.132896 | direct W13 | 0.020410 |
| activation | 0.010661 | retained by staged | 0.010661 |
| W2 | 0.053773 | direct W2 plus reduce | 0.016491 |
| exact reduce | 0.007478 | included above | 0 |

Both candidates exceeded the predefined `1.5x` fixed and `1.25x` routed
gates. No launch-parameter scan was needed.

## Numerics

All fixed and sequence outputs were finite, but neither candidate was
bit-exact because the custom GEMV reduction order differs from vendor GEMM.

| Candidate | Fixed max abs | 500-step finite | Sequence max abs | Mean abs |
| --- | ---: | ---: | ---: | ---: |
| Staged | 0.00012207 | 500/500 | 0.00012207 | 6.7841e-06 |
| Fused activation | 0.00012207 | 500/500 | 0.00024414 | 1.5606e-05 |

The fused activation intermediate reached `0.001953125` max absolute error,
while direct W13 reached `0.00024414`. The staged candidate is therefore the
only production candidate despite being about 10% slower on the routed
microbenchmark. It retains the vendor activation and has lower endpoint drift.

## Projection

Staged saves `0.218914 ms/layer` in this fixed-shape benchmark. If all 40 MoE
layers realize that saving, the static projection is about `8.76 ms/token`.
Applied mechanically to the qualified `15.8206 TPS` baseline, that corresponds
to roughly `18.4 TPS`. This is an upper-bound planning estimate, not a measured
service result, and it still falls short of 20 TPS by about 9%.

## Decision

`PASS SINGLE-GPU ALGORITHM GATE` for staged only. Next gates are:

1. reproduce timing and error bounds on physical GPUs 2 and 3;
2. add a default-off, strictly guarded production extension and unit tests;
3. wait for a four-GPU-qualified host before token-hash, multimodal, long
   decode, 262K startup, and strict service A/B.

Do not integrate the activation-fused variant and do not tune launch geometry.

Remote artifacts:

```text
/root/E_MOE_20/result.json
/root/E_MOE_20/bench.log
/root/E_MOE_20/smoke.json
/root/E_MOE_20/smoke.log
/root/E_MOE_20/logs/build_*.log
```
