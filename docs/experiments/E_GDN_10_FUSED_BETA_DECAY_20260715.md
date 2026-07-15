# E-GDN-10: Exact fused beta and decay preparation

## Scope

The TP4 rank-local decode profile measured the small beta/decay preparation
as a launch-dominated boundary. The original path launches separate operations
for FP16 beta sigmoid, FP32 conversion, `A_log` exponential, FP32 decay input
conversion/addition, softplus, multiplication, and the final recurrent decay
exponential. E-GDN-10 emits the recurrent step's FP32 beta and decay factor in
one CoreX kernel. Prefill and recurrent arithmetic are unchanged.

The fixed command uses `max_num_seqs=1`. The production hook additionally
requires contiguous FP16 `b_all/a_all/A_log/dt_bias`; unsupported dtypes or
layouts use the original PyTorch path. The explicit opt-out is
`BI100_GDN_COREX_BETA_DECAY=0`.

## Checkpoint dtype correction

The checkpoint safetensors metadata is authoritative:

```text
linear_attn.A_log   BF16 -> runtime FP16 after model downcast
linear_attn.dt_bias BF16 -> runtime FP16 after model downcast
```

An initial FP32-parameter probe was discarded. The retained kernel accepts
FP16 parameters and reproduces the original explicit/promotion-to-FP32
operation points. The benchmark also obtains `b_all/a_all` by splitting a
real `(1, 3088)` merged-projection layout, proving both views are contiguous
under the fixed batch-one contract.

## Correctness and performance

The production source was rebuilt independently and run on physical GPU1 with
100 warmups, 1,000 iterations per trial, nine trials, and 1,000 random input
sets. Beta and decay were bit-exact for every input.

| Path | Median (ms) | P10 (ms) | P90 (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| PyTorch reference | 0.062703 | 0.062583 | 0.062754 | 1.000x |
| Fused CoreX | 0.007123 | 0.007109 | 0.007127 | 8.803x |

```text
one-step beta/decay max_abs: 0 / 0
random exact steps: 1000/1000 beta, 1000/1000 decay
production source sha256: 491867433d36acda43c9a05b5da0724992f940490722bcf8a6a0abdd0b33fa97
production .so sha256: 958ba0a3f532357009754e645ac49dc9b1f29a35d468c0512075a3b0e2359909
merged views contiguous: true / true
remote result: /root/competition/E_GDN_10/production/results/gpu1_merged_view.json
```

The measured saving is `0.055581 ms/GDN layer`, approximately
`1.67 ms/token` across 30 GDN layers. This remains a rank-local projection,
not a TP4 service result.

## Decision

`KEEP AS TP4 QUALIFICATION CANDIDATE`. The primitive exactness and performance
gates pass and the Docker build has a fail-closed runtime fallback. Require an
identical 1,000-token service hash with the environment switch on/off before
describing the projected saving as measured end-to-end gain.
