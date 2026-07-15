# E-MOE-10: Exact CoreX weighted expert reduction

## Scope

E-MOE-04 replaced the T=1 expert weighting and sum with GEMV. It improved the
full routed path by 5-9%, but changed FP16 accumulation and failed the
1,000-token output hash. E-MOE-10 keeps the original arithmetic contract:

1. Multiply each of eight expert outputs by its FP16 routing weight.
2. Round each product to FP16.
3. Convert the rounded products to FP32 and accumulate them.
4. Round the final result to FP16.

```text
candidate: a2da709
production exactness fix: d6ac803
host: ssh-a2d0a302.default.gpu.phanthy.com
```

The candidate applies only to the fixed T=1, top-8, FP16 path. Other shapes
and dtypes retain `(expert_out * weights.unsqueeze(-1)).sum(...)` behind the
same runtime fallback. The opt-out is `BI100_MOE_COREX_EXACT_REDUCE=0`.

## Numerical gate

Three accumulation variants were tested. FP16 accumulation failed all 1,000
random inputs with maximum absolute difference 0.00390625. Both FP32 orders
were exact; the production candidate uses serial FP32 accumulation.

On physical GPU1-3, the selected variant passed 1,000/1,000 random reduction
inputs and complete routed-path output checks with maximum difference 0.

An important production-build gate caught a compiler-semantic regression.
Simplifying the kernel to multiply and accumulate in one loop produced
0/1,000 exact outputs even though the source appeared equivalent. Retaining
the tested runtime `Mode` kernel prevents CoreX Clang from reordering the
half-product conversion. The final production extension again passes
1,000/1,000 exact.

## Performance

The real decode shapes are `expert_out=(8,2048)` and `weights=(8,)`, FP16.

| Run | Existing reduce (ms) | Candidate reduce (ms) | Existing full (ms) | Candidate full (ms) | Full speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU1 initial | 0.027458 | 0.007432 | 0.503132 | 0.473022 | 1.064x |
| GPU2 | 0.027395 | 0.007468 | 0.503402 | 0.475314 | 1.059x |
| GPU3 repeat | 0.027503 | 0.007427 | 0.503123 | 0.473753 | 1.062x |

The first GPU3 run reported 1.019x because candidate composition latency was
transiently high; an independent serial repeat was stable over nine trials at
1.062x. An independent GPU1 repeat reported 1.115x but included high baseline
trials, so the table retains the more conservative initial GPU1 result.

Remote evidence is intentionally untracked:

```text
/root/competition/bench_runs/20260715_E_MOE_10/gpu1.json
/root/competition/bench_runs/20260715_E_MOE_10/gpu2.json
/root/competition/bench_runs/20260715_E_MOE_10/gpu3_repeat.json
```

The representative full-path saving is approximately `0.029 ms/layer`, or
`1.17 ms/token` across 40 MoE layers. This projects to about 1.6% end-to-end
decode improvement in isolation.

## Production build gate

The final production source compiled and passed its independent 1,000-random
input test on physical GPU1 without installation into active vLLM:

```text
source sha256: 94056cf66b786241f8d38455d08fac839aa6333a216b0fa6d1210b811773150a
shared object sha256: a36d84f345f986f329b521c1586269b0d2864c1655a1b4c0edec7a2ee703f9c7
exact steps: 1000/1000
```

Local Python/shell checks, diff validation, and P0 static coverage pass 41/41.

## Decision

`KEEP AS TP4 QUALIFICATION CANDIDATE`. Keep production code on
`exp/E-MOE-10-corex-exact-reduce` until a healthy four-card host passes exact
startup, full smoke, 1,000-token hash, long-context equality, and paired
service benchmarks. Do not merge the model/build hook into integration yet.
