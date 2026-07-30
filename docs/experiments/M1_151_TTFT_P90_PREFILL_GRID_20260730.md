# M1-151 TTFT-P90 prefill operator grid

Date: 2026-07-30

Status: the fixed 8K-64K BI100 operator grid qualified on physical GPUs
1, 2 and 3. This authorizes the private M1-152 P90-oriented short TP4
continuation screen. It does not authorize L2 activation replay, a long-context
performance claim, a default selector, `computility-run.yaml`, `main`, or an
official-score claim.

## Why P90 needs a separate screen

The platform result for `503fa7c670b6172d9a3e2912166e78317f5e289f`
contained 631 successful requests. Their TTFT buckets were:

| Prompt bucket | Successful requests | Bucket P90 |
|---|---:|---:|
| below 6K | 97 | 3.955 s |
| 6K-16K | 407 | 8.963 s |
| 16K-32K | 58 | 35.225 s |
| 32K-64K | 32 | 87.385 s |
| 64K-128K | 33 | 163.622 s |
| 128K-256K | 4 | 524.407 s |

The global P90 rank lies immediately after the first 562 requests covered by
the buckets through 32K. It is therefore governed by the upper 16K-32K and
lower 32K-64K region, not by the four 128K-256K requests. This explains why a
235K-only improvement can be real without materially changing the submitted
P90.

The same platform commit produced TTFT P90 values of 27.488 seconds and
14.529 seconds in two submissions with similar mean Output TPS. The M1-152
screen is a paired development screen; it does not treat one platform run as
enough evidence for a small regression or gain.

## Identity

- source revision:
  `634dd45f7fd6f62804e9ab0d31b3f2c8baf2bbb6`;
- instance: `ssh-73ca29ba`;
- physical GPUs: 1, 2 and 3;
- candidate extension SHA-256:
  `f94ad8abb554c6b2eb1a972ad7198cc4d57d36a166afe3e9b869985dda543236`;
- kernel source SHA-256:
  `11c387e6012834fe634ffa8d038f7a4bf4ec19fa13ec23779ee1f414037e564b`;
- shape: FP16, head dimension 256, block size 16, GQA 4:1 and query length
  8,176;
- one warmup and three timed trials per cell.

The eight cases were executed in three waves. The physical-GPU rotation was
1/2/3, 1/2/3, then 1/2. Every paged case used a permuted physical block table.

## Result

| Total prompt position | Reference | Candidate | Speedup | Output rel-L2 | Max abs |
|---|---:|---:|---:|---:|---:|
| 8,176 | 79.525 ms | 39.827 ms | 1.997x | 4.723e-6 | 2.441e-4 |
| 16,368 | 153.178 ms | 71.667 ms | 2.137x | 5.517e-6 | 3.052e-5 |
| 24,560 | 227.527 ms | 103.541 ms | 2.197x | 5.659e-6 | 3.052e-5 |
| 32,752 | 305.825 ms | 135.320 ms | 2.260x | 5.770e-6 | 3.052e-5 |
| 40,944 | 376.577 ms | 167.221 ms | 2.252x | 5.713e-6 | 1.526e-5 |
| 49,136 | 455.118 ms | 198.923 ms | 2.288x | 6.150e-6 | 1.526e-5 |
| 57,328 | 524.494 ms | 230.599 ms | 2.274x | 5.957e-6 | 1.526e-5 |
| 65,520 | 605.447 ms | 262.121 ms | 2.310x | 6.193e-6 | 1.526e-5 |

All outputs and LSE values were finite. Every cell passed relative L2
`<=1e-5`, maximum absolute error `<=1e-3`, and speedup `>=1.2x`. Minimum
speedup was 1.997x and median speedup was 2.256x. Accumulating the eight
attention-only medians reduced 2,727.691 ms to 1,209.220 ms, a 55.67%
reduction. This cumulative value is an operator estimate, not a projected
service TTFT.

The complete runner wall time was 28.087 seconds. Before/after preflight passed
on all three cards with zero free-memory drop. Postflight, scoped cleanup,
source identity and the privacy-safe fatal scan all qualified; fatal, OOM,
segfault, timeout, Gloo reset and worker-loss counts were zero.

## Harness incidents

An earlier GPU0 preflight child from this session was found under exact PID,
process-group and start-time identity. It did not exit after a 60-second
`SIGTERM` grace period, so it was sent `SIGKILL`; it then became a PID-1-owned
zombie with no GPU context. GPU1/2/3 passed a fresh allocation and matmul
preflight before M1-151.

The first relaunch, M1-150, was rejected before any operator cell because the
launcher had pre-created the private run root. M1-151 used a separate launcher
record and let the runner create its own new mode-0700 root. M1-150 is not
candidate performance evidence.

## M1-152 continuation

M1-152 keeps the old M1-141 audit contract unchanged and adds a separate
development funnel:

- cold prompts at 8K, 16K, 24K, 32K, 48K and 64K;
- partial-prefix continuations at 16K, 32K, 48K and 64K with about 8K
  residual prefill;
- exact same-arm cold/warm and partial/warm output checks;
- explicit cold, partial and fully warm cache-accounting checks;
- candidate dispatch proof and control dispatch exclusion;
- separate cold median, partial median and uncached-tail P90 screens;
- cross-arm generated-output identity retained as diagnostic only.

The frozen contract is `quality/short_tp4_p90_pair.v2.json`. Passing M1-152
only authorizes long-context confirmation. Real activation replay, complete
TP4 capability, long-context capacity and final performance gates remain
independent requirements.

Privacy-safe evidence is under
`docs/experiments/evidence/M1_151_TTFT_P90_PREFILL_GRID_20260730`.
