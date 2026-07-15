# E-GDN-12: Exact normalized q/k head mapping

## Scope

The decode path originally repeats four local FP16 q/k heads to eight value
heads before applying the same per-head L2 normalization, FP32 conversion, and
query scale. E-GDN-09 proved that normalizing before repetition is bit-exact,
but native `repeat_interleave` after normalization was slower. E-GDN-12 keeps
the original PyTorch FP16 L2 reductions on four heads and fuses only the exact
4-to-8 mapping, FP32 conversion, and query scaling into one CoreX kernel.

Recurrent state decay, both BMMs, rank-one state update, prefill, and weight
layout are unchanged. The production hook requires contiguous FP16 decode
views and head dimension 128; unsupported inputs use the original path. The
explicit opt-out is `BI100_GDN_COREX_QK_MAP=0`.

## Correctness and performance

The production source was independently rebuilt. The final benchmark obtains
q/k by splitting and reshaping the real `(1, 1, 2048)` convolution output;
both views are contiguous under the fixed batch-one evaluator contract.
Physical GPU1 ran 50 warmups, 500 iterations per trial, nine trials, and a
1,000-step random recurrent-state sequence.

| Boundary | Reference (ms) | Candidate (ms) | Speedup |
| --- | ---: | ---: | ---: |
| q/k prep | 0.132922 | 0.106459 | 1.2486x |
| prep + recurrent update | 0.193621 | 0.167567 | 1.1555x |

```text
one-step q/k/output/state max_abs: 0 / 0 / 0 / 0
random exact steps: 1000/1000
random output/state max_abs: 0 / 0
q/k split views contiguous: true / true
production source sha256: 5fc98479a039b1bb8331f3b7c6ffdb63699aefccc9020eeca9c854d0dc057b7d
production .so sha256: 774729ed1363f4b096ab4e1244e39b043e84c8fa44692e9418494dbbde92080a
remote result: /root/competition/E_GDN_12/production/results/gpu1_split_view.json
```

The conservative full-boundary saving is `0.026054 ms/GDN layer`, or about
`0.78 ms/token` across 30 GDN layers.

## Decision

`KEEP AS TP4 QUALIFICATION CANDIDATE`. The candidate passes bit-exact
primitive, 1,000-step state, real-layout, build, and performance gates. It
still requires on/off 1,000-token service hash equality and paired TP4 service
benchmarks before the projected saving can be called an end-to-end result.
