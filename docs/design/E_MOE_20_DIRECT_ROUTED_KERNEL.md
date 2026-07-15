# E-MOE-20 direct routed-expert kernel

## Problem

The current T=1 path spends about 70% of routed-expert time in three connected
regions: selected-weight gather, W13, and W2. Previous experiments optimized
each wrapper independently, but retained 12 MiB of selected-weight copying per
layer and token. The rejected pointer-batched experiment also retained general
BLAS dispatches, so it did not test a shape-specific direct kernel.

The rank-local decode shape is fixed:

```text
experts=256, top_k=8, hidden=2048, intermediate=128, dtype=FP16
```

## Algorithm

The prototype has two GPU stages:

1. Read the eight selected W13 matrices directly from the original expert
   tensor. One warp computes each output row. A paired variant computes gate
   and up together, reuses the input loads, and applies SiLU-and-multiply.
2. Read the selected W2 matrices directly. One warp owns one hidden output,
   iterates the eight experts in routing order, and performs the routed-weight
   reduction without materializing the `(8, 2048)` expert output.

The staged variant keeps the existing activation kernel between the two custom
stages. It isolates direct-addressing and W2-reduction value. The paired
variant additionally tests activation fusion.

This removes the 12 MiB gather and the `(8, 2048)` W2 output. It is materially
different from wrapping eight GEMVs: expert selection, matrix-vector work, and
the final reduction are part of the kernel schedule.

## Gates

The single-GPU prototype must satisfy all of the following before production
integration is considered:

- at least 1.5x on the fixed routed-expert boundary;
- at least 1.25x after routing is included;
- finite output for every random-sequence step;
- bounded FP16 error with no outlier growth across at least 500 steps;
- consistent results on every healthy physical GPU available.

Bit equality of intermediate W13 values is diagnostic, not the final quality
contract. A production candidate must later pass deterministic service token
hashes, long decode, multimodal smoke, score-relevant request success, and TP4
A/B. That qualification is deferred until a host passes all four independent
GPU preflights.

## Stop rule

Stop this route if the direct two-stage boundary misses the speed gates. Do not
scan launch parameters around a weak algorithm. If only the paired activation
variant fails numerically, retain the staged architecture and reject that
fusion alone.
