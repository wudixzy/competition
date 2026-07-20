# M1-34 Direct Single-Token Replay Guard - 2026-07-21

## Problem

M1-32 showed that `admission64/direct` can retain substantially more useful
prefix state than `fine32/direct`, but the original 10,593-token pressure
request was not deterministic after restoring 10,592 tokens. Its cold message
SHA-256 was `bc4f55ab...85b3`; after 17 competing sessions the request reported
a 10,592-token effective hit but returned `eba366dd...1e65`.

M1-33 aligned restores to the native 64-token GDN chunk and fixed correctness,
but replaying 48 extra tokens on every matrix warm request reduced Cache TPS
from `7607.9233` to `4122.5672`. Its weighted proxy was only `5126.1785`, so
coarse alignment is not a viable scoring policy.

## Diagnosis

Fresh pressure prompts at suffix lengths one and two both happened to produce
equal cold/warm outputs. Replaying the exact M1-32 run id reproduced the old
suffix-one failure, proving that the issue is prompt-sensitive numerical
divergence rather than stale binaries or nondeterministic state eviction.

The runtime has a concrete execution boundary at `T == 1`:

- routed MoE selects its optimized single-token direct kernel only for
  `T == 1`; `T >= 2` uses the grouped prefill implementation;
- attention head RMS normalization also has a shape-one specialization;
- the cold request computes its final token inside a multi-token prefill, while
  a 10,592-token restore leaves a physical one-token prefill.

A small arithmetic difference can therefore flip a near-tied first generated
token for some prompts. This explains why changing the prompt hid the failure
and why a full 64-token alignment was unnecessarily expensive.

## Fix

Direct restore now requires at least two physical prefill tokens. If the
longest strict 16-token block boundary leaves one token, capture falls back by
one block and replays 17 tokens. Suffix lengths 2 through 16 are unchanged.

Both sides of the scheduler contract enforce the rule:

- final capture uses `final_capture_key` for `fine32` and `admission64`;
- resident content keys are filtered by `restore_key_is_eligible` before the
  scheduler selects a restore;
- workers still execute only explicit scheduler actions and fail if a selected
  content-keyed state is absent.

No YAML parameter, cache capacity, default policy, or decode kernel changed.
`fine32/direct` remains the submission default while qualification is pending.

## Exact Regression Result

Repository and installed CoreX copies of `gdn_prefix.py` and `scheduler.py`
had identical SHA-256 values before the run. Runtime preflight confirmed:

- 10,593 tokens capture block 661, or 10,576 tokens;
- 10,594 tokens capture block 662, or 10,592 tokens;
- block 662 is ineligible for a 10,593-token direct restore.

The exact old run id `m1_32_admission64`, with the same suffix-one prompt and
17-session pressure sequence, then produced:

| Observation | Cached | Completion | Elapsed | Message SHA-256 |
| --- | ---: | ---: | ---: | --- |
| cold | 0 | 16 | 15.229 s | `bc4f55ab...85b3` |
| after pressure | 10,576 | 16 | 2.259 s | `bc4f55ab...85b3` |
| after refresh | 10,576 | 16 | 2.502 s | `bc4f55ab...85b3` |

`startup.rc`, `runtime_contract.rc`, `pressure.rc`, `fatal_scan.rc`, and the
aggregate `probe.rc` are all zero. The fatal scan is empty; no CUDA, OOM,
Gloo, worker-loss, or process-fatal event occurred. Cache trace output was not
enabled for this performance-neutral correctness run, so the API-reported
10,576 effective tokens are the persisted boundary evidence.

## Qualification Harness

`scripts/run_m1_34_fixed_matrix.sh` is fail-closed on the exact regression
artifacts. It additionally asserts the 10,593/10,576 token contract and three
equal hashes before it can start a fresh `admission64/direct` service. It then
runs startup capacity, smoke, the fixed 18-request `m1_32_ab` matrix, summary,
and the predeclared comparison gates.

Only a matrix-qualified candidate may run
`scripts/run_m1_34_post_matrix_gates.sh`, which checks 131K/256,
235K/1,000, and 262K/16 exact cold/warm replay plus 256K startup capacity and
fatal-log cleanliness.

Local discovery passes 223 tests with 24 optional-dependency skips. Submission
preflight passes 8/8, including the new scripts in shell syntax and LF checks.

## Fixed Matrix Result

The guarded direct candidate completed all 18 fixed-kernel requests with the
same `m1_32_ab` contract as the frozen baseline:

| Metric | `fine32/direct` | guarded `admission64/direct` | Delta |
| --- | ---: | ---: | ---: |
| Success | 100% | 100% | 0 pp |
| Effective cache hit | 49.9301% | 61.0671% | +11.1370 pp |
| Output TPS P10 | 21.6563 | 21.3347 | -1.49% |
| Input TPS | 741.4479 | 841.9203 | +13.55% |
| Cache TPS | 7,607.9233 | 7,437.7376 | -2.24% |
| TTFT P90, all | 20.8748 s | 18.2191 s | -12.72% |
| TTFT P90, warm | 1.4438 s | 1.4406 s | -0.23% |
| Weighted proxy | 6,699.4888 | 6,880.0051 | +2.69% |

All request-contract, success, hit-rate, and Output TPS gates passed. The
weighted-score gain was only `2.6945%`, below the predeclared `5%` stage gate,
so `compare.rc=1` and `qualification.rc=1`. The post-matrix long-context and
capacity script correctly did not run.

The score decomposition is more informative than the aggregate alone. Cold
TTFT fell by `13.469s` across nine requests and added about `281.2` weighted
points. Warm TTFT rose by `0.251s` in total, costing about `95.3` Cache-TPS
points; Output P10 variation cost another `5.4`. Passing the stage would
require roughly `0.401s` less total warm time, or about `45ms` per warm
request, with the other measurements fixed.

## Current Decision

Status: `CORRECTNESS_PASSED; PERFORMANCE_REJECTED`.

The narrow guard repairs the known direct correctness failure and preserves
the useful hit-rate gain, but M1-34 is not promotable under its declared score
gate. The next isolated experiment is not a parameter scan: `admission64`
currently recaptures and copies a final state back to CPU even when the same
content key is already resident and was just restored. Retaining the first
cold-captured canonical state should remove redundant warm-path transfer and
also avoid replacing it with a replay-derived state. This must be tested on a
new branch against the same fixed matrix; M1-34 remains closed evidence.
