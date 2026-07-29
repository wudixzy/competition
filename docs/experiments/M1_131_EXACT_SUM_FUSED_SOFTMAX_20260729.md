# M1-131 exact-sum fused softmax

## Scope

M1-131 is a single-component experiment for the fixed BI100 paged-prefill
shape:

- FP16 inputs and FP32 QK, softmax state, PV, and merge;
- head dimension 256;
- block size 16;
- four query heads to one KV head;
- query chunks no larger than 8192 tokens;
- total sequence length no larger than 262144 tokens.

It does not change decode, request semantics, model weights, sampling, cache
accounting, `computility-run.yaml`, or any unsupported-shape fallback.

## Motivation

M1-109 removed the separate maximum, normalization, split-sum, and state-sum
operations. Its component speedup was large, but later-token outputs diverged
from the M1-108 exact-output control on some full-model requests. The fused
kernel changed the FP32 additive reduction order used for each split sum.

M1-131 tests one narrower hypothesis. It fuses only:

1. the maximum reduction for each 512-token split;
2. the scan against the preceding online-softmax maximum;
3. correction generation; and
4. elementwise score normalization.

The authoritative M1-108 `at::sum(active_scores, -1)` operation and
`merge_split_sums_kernel` order remain unchanged. The fused kernel never
updates `running_sum`. Max reduction does not alter the selected finite value
for this NaN-free input contract, while explicit round-to-nearest subtraction
is used before `expf`.

This preserves the M1-108 sum and state-merge algorithm, not a static
bit-for-bit guarantee. The custom `fmaxf` tree and `expf` implementation still
have to prove exact output on BI100, followed by full-request TP4 validation.

The M1-109 source remains frozen at SHA-256
`11c387e6012834fe634ffa8d038f7a4bf4ec19fa13ec23779ee1f414037e564b`.
M1-131 uses a separate source, build script, module name, and shared object, so
it cannot overwrite either M1-108 or M1-109.

## Component gate

The control is the previously qualified M1-108 extension. Each of four healthy
BI100 cards runs one fixed production case:

| case | context | query |
|---|---:|---:|
| dense | 0 | 8176 |
| 65K | 65536 | 8176 |
| 128K | 122880 | 8176 |
| 235K | 229376 | 5616 |

Each process loads both extensions and alternates them on the same seeded
tensors. Reports contain only artifact identity, shape, bounded numerical
metrics, exactness booleans, and timings. They contain no tensor values,
prompts, token IDs, or model outputs.

The candidate must satisfy all of the following:

- both control and candidate outputs are finite;
- candidate output and LSE are exactly equal to M1-108;
- a repeated candidate call is exactly deterministic;
- output and LSE relative L2 error are at most `1e-5`;
- maximum absolute error is at most `1e-3`;
- median control/candidate speedup is at least `1.10x`;
- at least three of four shapes improve;
- no shape regresses by more than 2%;
- preflight, fatal scan, scoped cleanup, postflight, and repeated preflight all
  pass.

The exact equality requirement is intentionally stronger than the generic
custom-kernel numerical gate because this route exists to recover M1-109
performance without inheriting its output drift.

## Promotion boundary

A passing component result authorizes only a TP4 service experiment. It does
not authorize a default runtime change, `main` merge, official score claim, or
submission YAML change. TP4 still must pass next-token, complete-request,
functional, Agent, long-context, cache, performance, and lifecycle gates.

If either this implementation or one reasonable exact-order alternative fails
to provide a clear end-to-end gain, the route stops. M1-131 does not authorize
tile, threshold, stream, or YAML scans.
