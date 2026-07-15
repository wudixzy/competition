# E-GDN-03: Fused decode causal convolution update

## Scope

The Gated DeltaNet decode path previously used separate operations for the
three-token state concatenation, state update, four-tap depthwise convolution,
and SiLU activation. E-GDN-03 fuses that boundary into one CoreX kernel while
retaining the PyTorch path behind `BI100_GDN_COREX_CAUSAL_CONV=0`.

```text
prototype: 2085e51
exact FP16 rounding: 3a1a458
checkpoint-shape correction: 6d7edff
host: ssh-a2d0a302.default.gpu.phanthy.com
```

The authoritative checkpoint configuration is:

```text
hidden_size=2048
linear key heads=16, key head dim=128
linear value heads=32, value head dim=128
linear layers=30 of 40
```

Therefore the fixed-TP4 rank shape is `(batch=1, channels=2048, state=3)`,
because `(2048 q + 2048 k + 4096 v) / 4 = 2048`. An initial 2,560-channel
probe followed stale source comments and is diagnostic only; all decision
numbers below use 2,048 channels.

The runtime uses FP16 projection inputs and convolution weights but allocates
the Mamba convolution state as FP32. The fused kernel reproduces the reference
casts: each FP32 state value is rounded to FP16 for convolution, and the
convolution result is rounded to FP16 before SiLU. The latter intermediate
rounding changed the initial close-only result into bit-exact output without
adding a launch.

## Primitive gate

Each physical GPU ran nine serial timing trials after warmup and a 1,000-step
random sequence parity test. Candidate output and mutated FP32 state were
bit-exact on every card.

| Physical GPU | Reference median (ms) | Candidate median (ms) | Speedup | Output/state |
| --- | ---: | ---: | ---: | --- |
| GPU1 | 0.052086 | 0.007129 | 7.31x | exact |
| GPU2 | 0.052648 | 0.007125 | 7.39x | exact |
| GPU3 | 0.052017 | 0.007121 | 7.30x | exact |

Remote evidence is intentionally untracked:

```text
/root/competition/bench_runs/20260715_E_GDN_03/gpu1_actual2048.json
/root/competition/bench_runs/20260715_E_GDN_03/gpu2_actual2048.json
/root/competition/bench_runs/20260715_E_GDN_03/gpu3_actual2048.json
```

The median absolute saving is about `0.04496 ms` per GDN layer, or about
`1.349 ms/token` across 30 layers. Against the qualified E-MOE-03 decode range,
this projects to roughly 1.8% end-to-end Output TPS improvement. It is useful
but cannot by itself close the gap from approximately 13.3-13.5 to 20 TPS.

## Runtime integration status

The Docker patch builds `corex_gdn_causal_conv.so` into the discovered vLLM
package, and the model imports it with an explicit environment-controlled
fallback. Local P0 static coverage passes 41/41. The exact production source
compiled and imported on physical GPU1:

```text
source sha256: 877bc0633daf0ef25d605e144d71781a92e576dd8d2c286a97323ebad0984aeb
shared object sha256: d4b10a63662246063193e76bfa5f28cf88715250a2a1b117603c82be712ed42f
```

TP2 service diagnostics on physical GPU1+GPU2 failed during weight loading,
both without offload and with `cpu_offload_gb=8`; neither run reached model
forward or a GPU-block allocation. This is a model-capacity limitation and is
not evidence against the kernel. TP4 service hash, full smoke, 1,000-token
decode, long-context, and paired performance gates remain pending because
physical GPU0 is still unusable.

## Decision

`KEEP AS TP4 QUALIFICATION CANDIDATE`. The primitive is bit-exact and far above
its performance gate, but it must remain on
`exp/E-GDN-03-fused-causal-conv` until a healthy four-card runtime passes the
service gates. Do not merge it into `integration/perf-winners` or describe the
projected 1.8% as a measured end-to-end gain.
