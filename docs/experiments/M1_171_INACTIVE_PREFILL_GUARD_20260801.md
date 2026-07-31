# M1-171 inactive prefill guard short circuit

## Status

The private experiment branch restores the `503fa7c` short-circuit behavior
for the fused-prefill segment guard. When both fused prefill and activation
capture are disabled, the runtime no longer evaluates the candidate-only
shape, dtype, device, and contiguity contract for every full-attention
segment. Enabling either feature preserves the complete fail-closed contract.

This is a low-risk micro-optimization. It does not change model math, cache
policy, request semantics, context capacity, YAML, or `main`.

## Measurement

The removed guard was measured on `ssh-73ca29ba`, physical GPU 1, using the
qualified M1-170 CoreX runtime overlay. One hundred thousand calls on real
CoreX CUDA tensors took a net `9494.44323 ns` per call after subtracting the
empty-loop cost. A deliberately conservative 64-call request bound is only
`0.607644 ms`.

The platform short buckets regressed by seconds, not sub-milliseconds. This
guard therefore cannot explain the `fb0084f` platform regression and does not
justify a service or TP4 A/B. The branch keeps the short circuit because it
removes unintended disabled-path work, but no score or TTFT gain is claimed.

## Verification

- the focused paged-attention unit suite passed 26 tests;
- the complete tracked suite passed 1485 tests with 13 skips;
- submission preflight passed;
- the unit test proves the candidate contract is not called when both
  features are disabled;
- M1-164 untracked draft tests were excluded from the tracked-suite command.

Privacy-safe evidence is in
`docs/experiments/evidence/M1_171_INACTIVE_PREFILL_GUARD_20260801`.
