# M1-168 platform fb0084f triage

## Result identity

The supplied platform result belongs to
`fb0084fc778e62c26d6a6e108b87dc027ae2ed79`. It is compared only with the
same-shape `503fa7c670b6172d9a3e2912166e78317f5e289f` result. The formal
`computility-run.yaml` is byte-identical between the revisions. The supplied
result excerpt is not treated as a result for the newer experiment branch.

## What regressed

The request failure population did not change: both runs report 631/881 HTTP
200 responses, 250 errors, 226 tool-related 4xx, 22 image-related 4xx, seven
multi-system 4xx, and exactly 5,486,608 cached tokens. Functional pass rate
also remains 50/52, with `n=2` and base64 multimodal still failing. This is
not evidence of a newly introduced protocol regression.

Performance did regress in this sample:

| Metric | `503fa7c` | `fb0084f` | Change |
| --- | ---: | ---: | ---: |
| TTFT P90 | 27.488 s | 30.805 s | +12.07% |
| Output TPS P10 | 4.42 | 3.81 | -13.80% |
| Output TPS mean | 12.90 | 11.88 | -7.91% |
| Input TPS mean | 2018.60 | 1602.62 | -20.61% |
| Wall time | 34755.65 s | 36442.74 s | +4.85% |

The TTFT degradation is concentrated in shorter buckets: P90 is +39.04% for
inputs below 6K, +34.33% for 6K-16K, +11.01% for 16K-32K, and only about
1%-3% above 32K. This resembles added per-request or per-prefill-step overhead
more than a long-attention kernel regression.

The main default-path change after `503fa7c` is the admission64 GDN
capture-boundary correctness fix. It can add a small final prefill step so
that the recurrent state is captured exactly at a full KV block boundary.
That mechanism is a credible explanation for the short-request pattern, but
this platform pair does not prove causality because run-to-run platform
variance was already large. It must be tested by same-runtime TP4 A/B; the
correctness fix must not simply be removed.

## M1-170 follow-up

M1-170 subsequently isolated admission64 capture work with true-cold TP1
requests at 4K, 7.8K, and 16K. Both `admission64` and diagnostic `off` arms
reported zero cached tokens, and the A/B was repeated in both execution
orders. The geometric order-balanced admission64 overhead was `2.33%` for
the overall median and `1.72%` for P90. The 4K cell retained a positive
`7.54%` signal, while the 7.8K and 16K cells were about `2.03%` and `1.72%`.

This TP1 result does not establish TP4 platform causality, but it makes GDN
capture alone an implausible explanation for the observed `34%-39%` short
bucket regression. Admission cannot be disabled on this evidence because the
`off` arm is diagnostic and removes valid cache reuse. Further platform
regression work should profile request-level fixed overhead and first obtain a
complete, protocol-compatible 881-request population. M1-170 evidence is in
`docs/experiments/evidence/M1_170_COLD_CAPTURE_OVERHEAD_TP1_20260801`.

## 4xx attribution

The supplied excerpt contains only two 4xx markers and no matching 400 access
lines, so the v3 reconciliation correctly reports it as incomplete. The two
visible cases are:

- expected invalid `top_p` validation;
- one remote-image download that timed out after about five seconds.

The second case reached multimodal loading, but `str(TimeoutError())` was
empty, so the old marker became `unclassified_chat_error`. M1-168 now logs
only bounded `stage` and `exception_type` fields and maps the response to
`multimodal_load_failed`; it does not log the URL, image, messages, exception
text, or tool content. The summary retains schema v3 compatibility and adds
optional stage/type counters.

The 250 aggregate failures remain consistent with the previously identified
`max_completion_tokens` compatibility defect. `fb0084f` predates M1-165 and
does not declare that field. The current experiment branch includes the fix
and a qualified CoreX runtime probe, but this platform result cannot validate
it retroactively. A new submission is required to measure the success-rate
change.

## Qualification impact

Only 631 of 881 requests succeeded, for a 71.62% request success rate, and the
workload report records `timed_out=true`. The reported latency and throughput
distribution is therefore conditioned on the successful subset rather than
the required workload population. Using the documented formula with the
reported mean output TPS, mean input TPS, and cached TPM converted to tokens
per second gives a diagnostic linear value of about 4,769.58. This is not a
qualification score: fixing the 250 protocol failures changes the measured
request population and may materially move every percentile and throughput
term. The next platform run must establish protocol success before its
performance distribution can be compared with the hard targets.

## Boundaries

- M1-109 fused prefill remains disabled by default in both compared runs and
  cannot explain this result.
- The base64 functional failure is not the remote-image timeout shown in the
  excerpt; its exact response is still missing.
- The exact official `n=2` request shape is unknown, so the existing greedy
  sequential fan-out cannot yet be credited with fixing that test.
- The old run's 71.62% success rate and timeout make its performance metrics
  diagnostic only; they cannot be projected onto the current branch.
- M1-170 narrows the GDN-capture hypothesis but does not replace a TP4
  same-runtime causal A/B or a new platform submission.
- No formal YAML or `main` change is authorized by this triage.

Structured evidence is under
`docs/experiments/evidence/M1_168_PLATFORM_FB0084F_20260801`.
