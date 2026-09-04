# M1-178 FP16-QK attention-only teacher-forced distribution

## Decision

M1-162 shows obvious model-distribution drift in the two-arm quick screen and
requires reviewer adjudication. The experiment is valid, but its v2
distribution status is `distribution_drift_requires_adjudication / inconclusive`:
the adaptive protocol correctly stopped before control B, so no A/A-calibrated
formal pass or fail is available.

Across 256 fixed teacher-forced positions, control A and candidate agreed on
top-1 at 248 positions (96.875%). Eight top-1 flips occurred; five had a
control margin above the predeclared 0.1-nat quick-stop floor. Teacher-token
and shared-token absolute logprob delta P99 were 5.9475 and 4.3490 nats. This
is substantially more than a suffix-generation difference, but it is not
relabelled as an operator-numeric or task-capability failure.

The safe machine-readable summary is
`docs/experiments/evidence/M1_178_FP16_QK_TEACHER_FORCED_20260904/summary.json`.
Private per-position token identities and raw service logs remain only in the
remote `/tmp` experiment root.

## Adaptive workload

The full Qwen3.6-35B-A3B model ran in TP4 FP16 mode with block size 16 and
`max_model_len=262144`. Control A and candidate each used one service startup
and exactly four model requests at 4K, 16K, 32K and 64K. Every request sampled
64 positions, combining uniform coverage with every available 8192-token
chunk boundary. Four `/tokenize` calls per arm established cross-arm prompt
and teacher-token identity; they did not execute model inference.

All eight model requests returned HTTP 200 with complete usage,
`finish_reason=length`, one fixed greedy completion token and
`cached_tokens=0`. The non-streaming teacher-forced endpoint does not use SSE.
Candidate dispatch was observed four times; control dispatch was zero. Model,
tokenizer, command and all non-selector environment fields matched.

The old three-arm path would have issued 228 model requests: 72 unrelated
cache/protocol requests plus four teacher-forced requests for each arm.
M1-178 issued eight requests and skipped control B, eliminating 220 requests
(96.49%) and one of three service startups.

## Distribution observations

| Prompt tokens | Positions | Candidate minus control mean NLL |
| ---: | ---: | ---: |
| 4,096 | 64 | +0.189447 nats |
| 16,384 | 64 | +0.254248 nats |
| 32,768 | 64 | +0.010701 nats |
| 65,536 | 64 | -0.454518 nats |

The equal-cluster aggregate mean was -0.000031 nats, but the one-sided 95%
upper bound was +0.235650 nats and the per-length effects were strongly
heterogeneous. The severe-mean-NLL quick-stop rule (>0.05 nats aggregate) did
not trigger; the five high-margin flips did.

Bootstrap resampling treats each of the four length requests as a cluster and
then resamples positions within the selected cluster. It does not treat all
256 positions as independent. The summary explicitly records only four
independent clusters, so the interval is diagnostic and does not overstate
power. Control B and A/A noise were not collected because the earlier
high-margin stop condition fired.

Additional diagnostics were mutual top-k coverage 0.875, first divergent
sample ordinal 10 and control top-1 margin P99 13.375 nats. Top-1 agreement is
reported, not compared with a universal 98% threshold.

## Runtime and lifecycle

- evidence revision: `51f6ebeb1c55cdc5c27352d0dc00b3d1b9f5bf5d`;
- final harness revision before this report:
  `f51655b4e3e718f5bbacd2478d646715ae0768f9`;
- instance: `cc-ce19242b-436f-4141-868b-610eb3ac8cee-0`;
- model/tokenizer:
  `/root/public-storage/models/Qwen/Qwen3.6-35B-A3B`;
- runtime: CoreX 3.2.3, Python 3.10.12, Torch 2.1.0, vLLM 0.6.3,
  Transformers 4.55.3 and clang 16.0.6;
- reused four-card preflight:
  `m1-176-focus-preflight-4e2b2e7`;
- control A service wall time: 703.609 seconds;
- candidate service wall time: 682.506 seconds;
- total adaptive funnel wall time: 1442.069 seconds.

Both arms completed TERM-first scoped cleanup and qualified postflight. Final
API server, worker and GPU process counts were zero. Fatal scans found zero
OOM, segfault, non-finite state, collective reset, timeout or worker loss.

One earlier attempt is excluded as invalid. Control A completed, but the
candidate API server encountered `EADDRINUSE` during a closed-socket reuse
window before loading the model or issuing any request. Commit `51f6ebe`
replaced listener-only admission with a bounded wait for actual bindability;
the valid rerun crossed that boundary and completed both arms.

The evidence run enabled the existing verbose cache trace only to retain the
zero-cache observation. Its block-hash log was unnecessary and is not
committed. The final harness disables that trace because `cached_tokens` is
already read directly from response usage; this reporting-only reduction was
not used to reinterpret the measured logits.

## Layered conclusions

- Operator numerics: M1-176 G2 remains `pass`; no operator replay was rerun.
- Kernel timing: M1-176 synthetic timing remains the applicable evidence; no
  timing was rerun.
- TP4 service performance: M1-176/M1-177 gains are retained, not recomputed by
  this distribution workload.
- Distribution: valid quick-screen evidence shows obvious drift and is
  `inconclusive` pending adjudication; no formal A/A envelope exists.
- Capability: not run, so no capability conclusion exists.
- Promotion: not authorized. No main, YAML, default-selector, 881-request,
  235K teacher-forced, cache matrix or full capability work was performed.

Work stops here for reviewer review. A small capability non-inferiority screen
is not automatically authorized because the formal A/A-calibrated
distribution gate did not pass.
