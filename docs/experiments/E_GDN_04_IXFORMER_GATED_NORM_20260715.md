# E-GDN-04: ixformer RMSNorm in the GDN decode tail

## Scope

E-GDN-04 tests whether the installed ixformer RMSNorm can replace the
pure-PyTorch normalization inside `Qwen3_5RMSNormGated`. The benchmark uses
the authoritative TP4 rank shape and keeps the same actual vLLM linear layer
after both normalization paths:

```text
core state: (8, 128), FP32
gate:       (8, 128), FP16
weight:     (128,), FP16 runtime dtype
out input:  (1, 1024), FP16
out output: (1, 2048), FP16
```

The checkpoint stores BF16 weights, but the vendor runtime logs show that the
model is downcast to FP16. The Mamba/GDN state remains FP32.

## Capability and result

The installed `ixformer.functions.rms_norm.rms_norm` maps to
`ixformer_torch_ops::rms_norm_forward`. No gated RMSNorm operator is present.

On physical GPU1, the FP32 input variant failed at the operator contract:

```text
input type must be Half/BFloat16
```

Casting the state to FP16 allowed the operator to run, but changed the result:

| Path | Norm median (ms) | Full tail median (ms) | Tail speedup | Norm max abs | Tail max abs | Close |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch FP32 reference | 0.09242 | 0.10776 | 1.000x | 0 | 0 | yes |
| ixformer FP16 | 0.04777 | 0.06037 | 1.785x | 0.00390625 | 0.0009765625 | no |

Evidence is intentionally untracked:

```text
/root/competition/bench_runs/20260715_E_GDN_04/gpu1.json
/root/competition/bench_runs/20260715_E_GDN_04/gpu1.log
```

## Decision

`REJECT AS PERFORMANCE WINNER`. The available vendor operator does not accept
the FP32 state, and a pre-operator FP16 cast fails the primitive numerical
gate. Do not integrate ixformer RMSNorm or change the GDN state dtype. A custom
kernel is justified only if it preserves the reference FP32 reduction and
final FP16 output semantics.
