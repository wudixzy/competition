# E-GDN-09: Eight-head GDN candidate audit

## Scope

E-GDN-08 verified from the checkpoint `config.json` that TP4 ranks have eight
local value heads, not twelve. E-GDN-09 reruns the two exact candidates whose
earlier rejection evidence used the stale twelve-head assumption. Production
code is not changed unless a candidate passes both exactness and performance
gates.

Both probes ran serially on physical GPU1 with 500 iterations per trial and
nine trials. The recurrent extension was rebuilt from the tracked test source:

```text
extension sha256: 8606d578b25cf3077299d5d87f0e0d20ce29b0a78d2fb11de7c525670f6a04de
norm result: /root/competition/E_GDN_09/results/norm_before_expand_gpu1.json
recurrent result: /root/competition/E_GDN_09/results/recurrent_gpu1.json
```

## A: Normalize before expansion

This comparison includes q/k normalization and expansion, state clone and
decay, two BMMs, delta construction, and the in-place rank-one update.

| Path | Median (ms) | P10 (ms) | P90 (ms) | Speedup | Output/state |
| --- | ---: | ---: | ---: | ---: | --- |
| Normalize after expansion | 0.199526 | 0.198842 | 0.200916 | 1.0000x | exact |
| Normalize before expansion | 0.202565 | 0.202369 | 0.203024 | 0.9850x | exact |

The reorder remains bit-exact but regresses by 1.50%.

## B: Exact recurrent extension

Both paths execute identical production q/k expansion, FP16 L2 normalization,
FP32 conversion, and query scaling. The candidate replaces only state decay,
memory BMM, delta, rank-one update, and output BMM with the preallocated CoreX
kernel. This avoids the later inverse-normalization prototype, which is a
different and already rejected numerical path.

| Path | Median (ms) | P10 (ms) | P90 (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| PyTorch full prep + recurrent | 0.210085 | 0.209653 | 0.210734 | 1.0000x |
| Prep + exact recurrent kernel | 0.213186 | 0.213131 | 0.213286 | 0.9855x |

The custom arithmetic is not bit-exact, but remains inside the established
1,000-random-step gate: output maximum absolute difference `2.24e-8`, state
maximum absolute difference `2.38e-7`, both close and finite. Performance
regresses by 1.45%.

## Decision

`REJECT BOTH`. Correcting the old shape assumption does not reverse either
decision. Keep production q/k normalization after expansion and the PyTorch
recurrent implementation. Do not spend TP4 service qualification time on
these candidates. The E-GDN-08 hotspot map remains the source for choosing a
different optimization boundary.
