# E-GDN-13: Current optimized GDN decode profile

## Scope

E-GDN-13 re-runs the complete TP4 rank-local decode layer after combining
E-GDN-03 causal convolution, E-GDN-05 gated norm, E-GDN-10 beta/decay, and
E-GDN-12 normalized q/k mapping. The reference keeps E-GDN-03/05 but disables
the E-GDN-10/12 operation replacements. Both paths use identical weights,
inputs, convolution state, temporal state, recurrent BMMs, and projections.
TP communication and output all-reduce remain excluded.

Physical GPU1 used 50 warmups, 500 iterations per trial, and nine trials.
All four production extensions were loaded from their independently rebuilt
artifacts.

## Combined result

| Rank-local layer | Median (ms) | Speedup | Saving (ms/layer) |
| --- | ---: | ---: | ---: |
| E-GDN-03/05 reference | 0.549146 | 1.000x | - |
| Current E-GDN-03/05/10/12 | 0.449503 | 1.2217x | 0.099642 |

The combined output, convolution state, and temporal state are bit-exact;
output and temporal-state maximum absolute differences are zero. The net
saving is approximately `2.99 ms/token` across 30 GDN layers. This is stronger
and more representative than adding isolated E-GDN-10/12 projections.

## Current stage map

| Current stage | Median (ms) | Isolated share |
| --- | ---: | ---: |
| Merged input projection | 0.133088 | 31.83% |
| Fused causal conv | 0.007273 | 1.74% |
| Fused beta/decay | 0.007248 | 1.73% |
| Exact normalized q/k map | 0.097710 | 23.37% |
| PyTorch recurrent update | 0.060277 | 14.42% |
| Fused gated norm | 0.040744 | 9.74% |
| Local output projection | 0.071803 | 17.17% |

The current isolated sum is `0.418143 ms`; composition overhead is
`0.031361 ms`. The full current layer is `0.449503 ms`.

```text
remote result: /root/competition/E_GDN_13/results/gpu1_v2.json
scope: one TP4 rank, no collectives
```

## Decision

`PROFILE COMPLETE`. Use the combined `2.99 ms/token` projection when updating
the candidate stack. Stop repeating these already-audited GDN directions:

- cuBLAS projection mode scans (E-GDN-11: below 1%);
- custom q/k normalization reductions (nonexact in E-GDN-06);
- exact custom recurrent kernel (regressed in E-GDN-09).

Further GDN work needs a new exact larger boundary. TP4 service/hash
qualification has higher priority once a healthy fourth GPU is available.
