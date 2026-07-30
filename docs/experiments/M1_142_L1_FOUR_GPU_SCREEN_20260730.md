# M1-142 four-GPU L1 operator screen

Date: 2026-07-30

Status: the frozen L1 synthetic operator screen passed on four independent
BI100 GPUs. This authorizes the M1-109 fused-prefill candidate to proceed to
L2 real-activation capture/replay. It does not establish TP4 end-to-end
performance, model capability, cache transparency, a production default,
`main` eligibility, or an official-score claim.

## Identity

- source revision:
  `96311e85abe14da07c4b23460c563415fe7d1d65`;
- instance: `ssh-73ca29ba`;
- candidate extension SHA-256:
  `f94ad8abb554c6b2eb1a972ad7198cc4d57d36a166afe3e9b869985dda543236`;
- kernel source SHA-256:
  `11c387e6012834fe634ffa8d038f7a4bf4ec19fa13ec23779ee1f414037e564b`;
- CoreX runtime identity: `corex-3.2.3-m1-142`;
- fixed shape assignment: one production shape on each physical GPU.

The candidate binary was rebuilt from the same M1-109 kernel source through
the content-addressed CoreX build cache. Binary identity differs from the
historical prebuilt bundle because this build used the active CoreX Python
toolchain; the source identity and output numerics reproduce the M1-109
candidate.

## Result

| Shape | GPU | Reference ms | Candidate ms | Speedup | Output rel-L2 | LSE rel-L2 | Max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense q8176 | 0 | 78.769 | 39.804 | 1.979x | 4.723e-6 | 1.910e-8 | 2.441e-4 |
| 65K q8176 | 1 | 674.953 | 293.945 | 2.296x | 6.174e-6 | 1.533e-8 | 1.526e-5 |
| 128K q8176 | 2 | 1209.055 | 515.608 | 2.345x | 6.054e-6 | 1.508e-8 | 7.629e-6 |
| 235K q5616 | 3 | 1546.805 | 658.473 | 2.349x | 6.625e-6 | 1.663e-8 | 7.629e-6 |

All candidate outputs and LSE values were finite. Every shape passed the
frozen relative-L2 limit of `1e-5`, max-absolute limit of `1e-3`, and minimum
speedup of `1.5x`. Minimum observed speedup was `1.979x`.

All four cells ran in one parallel wave. Cell wall time was `11.015s`; total
runner wall time including preflight, postflight and evidence generation was
`25.466s`. This is the intended low-cost L1 screen and did not start or load
the model service.

## Lifecycle

Four-card allocation and matrix-multiply preflight passed before and after the
screen. Every card reported `34,057,748,480` free bytes in both observations,
with zero free-memory drop. Scoped child cleanup reaped all processes. Final
postflight found no API server, worker, or GPU process.

The privacy-safe recursive fatal scan reported zero CUDA, CoreX, segfault,
OOM, timeout, Gloo-reset, and worker-loss markers. The source worktree
remained unchanged.

## Decision

`full_l1_contract_satisfied=true` and
`l2_capture_authorized=true`. The next experiment is one baseline TP4
activation-bank capture followed by four parallel rank replays under the
M1-138 calibrated numerical contract. L2 must pass before the short TP4
screen is allowed.

Retained evidence is under
`evidence/M1_142_L1_FOUR_GPU_20260730`. It contains only artifact identities,
fixed tensor shapes, timings, scalar numerical errors, GPU health and
lifecycle summaries. It contains no prompts, model outputs, activations,
token IDs, credentials, or model weights.
