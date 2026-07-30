# M1-144 to M1-146 three-GPU L1 replication

Date: 2026-07-30

Source revision:
`fb0084fc778e62c26d6a6e108b87dc027ae2ed79`

Instance: `ssh-73ca29ba`

## Purpose

GPU 0 remained unavailable, so the fixed M1-142 four-case operator matrix was
rotated across healthy GPUs 1, 2, and 3. This checks whether the M1-109
fused-prefill operator result depends on one physical card. It does not
substitute for the frozen four-GPU L1 contract, real activation replay, or a
full-model TP4 A/B.

All rotations used the same source, kernel, and candidate extension:

- kernel source SHA-256:
  `11c387e6012834fe634ffa8d038f7a4bf4ec19fa13ec23779ee1f414037e564b`;
- extension SHA-256:
  `f94ad8abb554c6b2eb1a972ad7198cc4d57d36a166afe3e9b869985dda543236`.

## Results

The GPU orders were:

- M1-144: `1,2,3`;
- M1-145: `2,3,1`;
- M1-146: `3,1,2`.

Every case in every rotation was finite and qualified. All lifecycle,
postflight, repeated preflight, fatal-scan, cleanup, and source-identity gates
passed.

| Case | Speedup range | Cross-card spread | Output relative L2 |
| --- | ---: | ---: | ---: |
| dense q8176 | 1.9797x-1.9819x | 0.11% | 4.7230e-6 |
| 65K q8176 | 2.2934x-2.3215x | 1.22% | 6.1741e-6 |
| 128K q8176 | 2.3071x-2.3449x | 1.62% | 6.0542e-6 |
| 235K q5616 | 2.2955x-2.2963x | 0.04% | 6.6247e-6 |

The LSE relative L2 range was 1.5084e-8 to 1.9101e-8. No NaN, Inf, fatal,
timeout, Gloo reset, worker loss, or unreaped run process was observed.

## Interpretation

The operator speedup and numeric profile reproduce across all three healthy
cards with low dispersion. This is sufficient to retain M1-109 as the active
candidate and to stop further single-card parameter scanning.

It is not sufficient to:

- claim the four-GPU L1 contract;
- authorize activation capture/replay L2;
- claim TP4 or end-to-end TTFT improvement;
- modify `computility-run.yaml`, `main`, or a production default;
- claim an official score.

When GPU 0 recovers, the next required work is the exact four-rank L1 rerun,
followed by hash-bound real activation replay and the paired short TP4 funnel.
The privacy-safe source reports are under
`docs/experiments/evidence/M1_144_146_THREE_GPU_L1_20260730`.
