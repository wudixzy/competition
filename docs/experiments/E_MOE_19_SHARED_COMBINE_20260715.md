# E-MOE-19: Fused shared-expert gate and routed add

## Hypothesis

The Qwen3.6 MoE tail applies a scalar sigmoid gate to the shared-expert output
and then adds the routed-expert output. At T=1, PyTorch launches separate
sigmoid, multiply, and add operators over only 2,048 FP16 elements. E-MOE-19
tested whether one shape-independent CoreX kernel could remove this launch
overhead without changing output bits.

## Exactness work

The first kernel used native half multiply and add. It reached `2.50x`, but
only 2 of 1,000 random cases were exact, with maximum absolute error
`0.00390625`. Retaining PyTorch sigmoid did not fix the mismatch.

Converting to FP32 and back was still reordered by the compiler. The final
kernel stores the FP16 product in a `volatile half`, forcing the same
intermediate rounding boundary as the two eager PyTorch operators. Its shared
object SHA256 is:

```text
26df49ceadd4a4fa0c6081bda248b552f56b008c4bda2e7d7ea4793736c88b40
```

Final correctness gates on GPU1:

```text
one step:                     exact, max_abs=0
random combine:               1,000/1,000 exact, max_abs=0
all finite FP16 gate patterns 63,488/63,488 exact, max_abs=0
full TP4 rank-local MoE:       100/100 exact, max_abs=0
```

## Performance

The isolated tail at shape `(1, 2048)` improved from `0.017842 ms` to
`0.007131 ms`, or `2.502x`.

The complete synthetic TP4 rank-local MoE boundary used the checkpoint shapes
of 256 experts, top-8 routing, routed intermediate 128, shared intermediate
128, and hidden size 2,048. It included the fused router/shared gate, E-MOE-13
weight gather, routed W13/W2, E-MOE-10 exact reduction, shared W13/W2, and the
final combine:

```text
reference full: 0.611500 ms/layer
candidate full: 0.601651 ms/layer
speedup:        1.01637x
saving:         0.009850 ms/layer
40-layer sum:   0.393991 ms/token
```

Raw remote artifacts:

```text
/root/competition/E_MOE_19/results/gpu1_v2.json
/root/competition/E_MOE_19/results/gpu1_v3.json
/root/competition/E_MOE_19/results/gpu1_v4.json
/root/competition/E_MOE_19/results/gpu1_v5.json
/root/competition/E_MOE_19/results/gpu1_full_v1.json
```

## Decision

`REJECT FOR PRODUCTION`. The final kernel is bit-exact and the isolated
operator is faster, but the complete MoE boundary improves only 1.64%, below
the established 5% integration gate. A projected `0.394 ms/token` does not
justify another compiled extension and an additional unqualified TP4 runtime
path. Keep the probe as evidence; do not change the current candidate stack.
