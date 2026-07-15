# E-GDN-05: CoreX gated norm output fusion

## Scope

E-GDN-04 proved that the installed ixformer RMSNorm cannot consume the FP32
GDN state. E-GDN-05 instead keeps the original PyTorch FP32 square/mean/rsqrt
reduction and fuses the remaining scale, FP16 weight, FP32 SiLU gate, multiply,
and FP16 output conversion into one CoreX kernel.

```text
candidate: d823dbd
host: ssh-a2d0a302.default.gpu.phanthy.com
```

The decode-only production hook is fail-closed. It requires FP32 state, FP16
gate and weight, and head dimension 128; every other dtype or shape uses the
original PyTorch method. Prefill is unchanged. The runtime opt-out is
`BI100_GDN_COREX_GATED_NORM=0`.

## Numerical development gate

Two custom reduction kernels were initially faster but rejected before model
integration:

| Reduction | Exact random steps | Norm max abs | Tail max abs |
| --- | ---: | ---: | ---: |
| parallel tree | 976/1000 | 0.0009765625 | 0.00048828125 |
| serial | 909/1000 | 0.0009765625 | 0.0009765625 |

The retained variant passes the exact PyTorch-computed inverse to the custom
output kernel. On all three healthy cards, 1,000/1,000 random inputs produced
bit-exact norm and actual vLLM linear-tail outputs.

## Performance

The real fixed-TP4 rank boundary is:

```text
core/gate: (8, 128)
normed out-projection input: (1, 1024)
out-projection output: (1, 2048)
```

Each result is from serial trials on one physical GPU. Both paths use the same
actual `ReplicatedLinear` runtime layer as the out-projection oracle.

| GPU | Reference tail (ms) | Candidate tail (ms) | Speedup | Random exact |
| --- | ---: | ---: | ---: | ---: |
| GPU1 | ~0.1075 | 0.053118 | 2.024x | 1000/1000 |
| GPU2 | 0.107836 | 0.053439 | 2.018x | 1000/1000 |
| GPU3 | 0.113936 | 0.057580 | 1.979x | 1000/1000 |

Remote evidence is intentionally untracked:

```text
/root/competition/bench_runs/20260715_E_GDN_05/gpu1_v2.json
/root/competition/bench_runs/20260715_E_GDN_05/gpu2_v2.json
/root/competition/bench_runs/20260715_E_GDN_05/gpu3_v2.json
```

The median saving is approximately `0.0544 ms` per GDN layer, or
`1.63 ms/token` across 30 layers. In isolation that projects to about 2.2%
end-to-end decode improvement. Combined with the separately measured
E-GDN-03 primitive saving, the unqualified projection is about `2.98 ms/token`
or roughly 4%; no combined service result exists yet.

## Production build gate

The production source compiled and executed its real mixed-dtype contract on
physical GPU1 without being installed into the active vLLM package:

```text
source sha256: 104cb0aba009dcc13e5a917e4c3b462231bad6ee6fa17bdf1b6f209824566310
shared object sha256: 70e6dd5bcb608fe43217314c4d3d4095d91c29c50a2bfa53cf7f8d1aff637a86
output shape/dtype: (8, 128), FP16, finite
```

Local Python/shell checks, diff validation, and P0 static coverage pass 41/41.

## Decision

`KEEP AS TP4 QUALIFICATION CANDIDATE`. The primitive and full-tail gates pass,
but the candidate remains on `exp/E-GDN-05-corex-gated-norm` until a healthy
four-card host completes exact startup, full smoke, 1,000-token greedy hash,
long-context cold/warm equality, and paired service benchmarks. Do not merge
the model or build hook into `integration/perf-winners` yet.
