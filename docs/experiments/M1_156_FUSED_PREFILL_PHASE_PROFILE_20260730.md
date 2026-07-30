# M1-156 fused-prefill phase profile

Date: 2026-07-30

Status: candidate-only CoreX phase attribution qualified on physical GPUs 1,
2, and 3. This authorizes one bounded implementation aimed at QK/PV data flow.
It does not authorize a TP4 service claim, long-context claim, default
selector, `computility-run.yaml`, `main`, or an official-score claim.

## Motivation

M1-151 proved that the M1-109 fused-prefill operator is about 2.0x-2.3x faster
than the PyTorch reference at the 8K-64K positions that influence submitted
TTFT P90. It did not identify which parts of the candidate still dominate.
M1-156 profiles the existing qualified binary rather than changing a tile or
threshold.

The fixed cases use query length 8,176 and total KV positions 16,368, 32,752,
and 65,520. These correspond to strict-prefix segments inside 16K, 32K, and
64K prompt chunks.

## Profiler correction

The first M1-155 invocation was invalid. CoreX ixprof ignored
`cudaProfilerStart/Stop` range filtering and captured one PyTorch reference,
one candidate warmup, and three candidate trials. All three workload cells and
all lifecycle checks passed, but launch counts and timing coverage correctly
failed qualification.

M1-156 avoids inferred subtraction:

- the profiled process invokes no reference implementation;
- only the candidate invokes the uniquely named split4 and GEMM kernels;
- one candidate warmup plus three timed candidate calls appear in ixprof;
- each candidate-owned phase is scaled by the fixed `3/4` ratio;
- one CUDA event range measures the three timed calls;
- all other process kernels remain diagnostic and are not assigned to the
  candidate;
- numerical correctness is hash-bound to the same extension's qualified
  M1-151 fixed-tensor evidence, while the current output is checked finite.

All expected launch counts matched exactly. Candidate-owned GPU activity
covered 98.05%, 98.70%, and 99.03% of the event ranges.

## Identity

- source revision:
  `4caffefba14aa5d9d519a7226d6fe71e0999893a`;
- instance: `ssh-73ca29ba`;
- physical GPUs: 1, 2, and 3;
- extension SHA-256:
  `f94ad8abb554c6b2eb1a972ad7198cc4d57d36a166afe3e9b869985dda543236`;
- kernel source SHA-256:
  `11c387e6012834fe634ffa8d038f7a4bf4ec19fa13ec23779ee1f414037e564b`;
- numerical lineage:
  `M1_151_TTFT_P90_PREFILL_GRID_20260730/runner_status.json`, SHA-256
  `4ad1c41ffb2d54ef86eb1bf0444012ccda7f0f9495969b273e26a364964cf9aa`.

## Result

| Phase | 16K | 32K | 64K |
|---|---:|---:|---:|
| QK GEMM | 31.92% | 34.02% | 35.20% |
| PV GEMM | 31.94% | 33.91% | 34.91% |
| Normalize | 18.58% | 19.71% | 20.31% |
| Causal mask | 9.97% | 5.29% | 2.73% |
| Merge output | 4.69% | 4.98% | 5.16% |
| Gather | 0.40% | 0.50% | 0.56% |
| Query conversion | 0.55% | 0.29% | 0.15% |
| Unattributed candidate | 1.95% | 1.30% | 0.97% |

QK plus PV rises from 63.86% at 16K to 70.12% at 64K. Gather remains below
0.6%, so optimizing physical-page reads alone has little remaining upside in
this implementation. Normalize is the next largest phase, but M1-109 already
fused it and its standalone upper bound is only about one fifth of the current
candidate.

The three-call CUDA event totals were 219.521, 413.506, and 801.551 ms, or
73.174, 137.835, and 267.184 ms per call. These are close to the unprofiled
M1-151 medians of 71.667, 135.320, and 262.121 ms; profiler overhead is not
used as a performance claim.

## Decision

The next bounded candidate should reduce QK/PV data movement or compute cost
while retaining FP32 accumulation. The first implementation will keep the
existing online-softmax and PV path, but test FP16 Q/K GEMM inputs with FP32
output/accumulation. This removes FP32 key expansion and tests whether CoreX
GEMM can exploit the narrower input path without changing model precision.

It must pass the same fixed-tensor finite, relative-L2, max-absolute, and
speed screens before any model startup. If CoreX rejects the mixed GEMM
contract or the numerical/performance gate fails, one deeper fused alternative
is allowed; this is not authorization for a parameter sweep.

All three cells, scoped cleanup, postflight, repeated preflight, free-memory
comparison, fatal scan, and source identity passed. Runner wall time was
15.680 seconds. Privacy-safe evidence is under
`docs/experiments/evidence/M1_156_FUSED_PREFILL_PHASE_PROFILE_20260730`.
