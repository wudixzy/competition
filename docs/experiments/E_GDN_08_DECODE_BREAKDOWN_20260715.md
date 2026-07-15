# E-GDN-08: TP4 rank-local decode breakdown

## Scope

This experiment profiles one Qwen3.6-35B-A3B Gated DeltaNet decode layer at
the fixed TP4 rank-local shape. It includes both local projections and the
current E-GDN-03/E-GDN-05 production candidates, but excludes TP communication
and `RowParallelLinear` all-reduce. It is a hotspot map, not an end-to-end TPS
measurement.

The checkpoint `config.json` is authoritative:

```text
hidden_size=2048
linear_num_value_heads=32
linear_num_key_heads=16
linear key/value head dimension=128
linear layers=30 of 40
```

Thus each TP4 rank uses eight value heads, four key heads, a 1,024-wide value
projection, a 2,048-wide q/k/v convolution, and a 3,088-wide merged input
projection. Older E-GDN-02/06/07 probes used twelve local value heads after
following stale source comments; their rejection decisions require an
eight-head audit before being treated as final.

## Method

Current production extension sources were rebuilt in an isolated remote
directory and loaded directly. Each stage used 50 warmups, 500 serial decode
steps per trial, and nine trials on physical GPU1. The benchmark reproduces
the production `_l2norm` operation order and mutates convolution and temporal
states in place.

```text
causal-conv .so sha256: 6dff5da871ebc2d463628454d28569249ab922625709e82e61f900f0508575a2
gated-norm .so sha256:  1ea3b0b3217a94d20ff269ce258b8983faa910fa0337d410254120ded67d8feb
remote result: /root/competition/E_GDN_08/results/gpu1_v2.json
```

## Result

| Rank-local stage | Median (ms) | P10 (ms) | P90 (ms) | Isolated share |
| --- | ---: | ---: | ---: | ---: |
| Merged input projection | 0.135359 | 0.135104 | 0.135480 | 25.97% |
| Fused causal conv | 0.007554 | 0.007534 | 0.009913 | 1.45% |
| Beta and decay | 0.043207 | 0.043151 | 0.043290 | 8.29% |
| Head expansion and q/k prep | 0.142524 | 0.142429 | 0.142628 | 27.34% |
| Recurrent state update | 0.075507 | 0.075329 | 0.076195 | 14.49% |
| Fused gated norm | 0.042872 | 0.042804 | 0.042968 | 8.23% |
| Local output projection | 0.074209 | 0.074164 | 0.074244 | 14.24% |

The isolated sum is `0.521232 ms`. The complete rank-local layer is
`0.564578 ms`, leaving `0.043346 ms` of Python composition/split/view overhead.
Across 30 GDN layers, the measured local boundary represents about
`16.94 ms/token`, still excluding TP collectives and the ten full-attention
layers.

## Decision

`PROFILE COMPLETE`. Do not optimize E-GDN-03 further: causal convolution is
only 1.45% of this boundary after fusion. Audit normalization-before-expansion
and the recurrent extension at the correct eight-head shape next. Input and
output projection work is also material, but E-GDN-01 has already merged the
input projections and projection-kernel changes require a stronger exactness
and integration case.
