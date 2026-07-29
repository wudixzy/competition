# M1-137 IFEval power149 fused-prefill A/B

Date: 2026-07-30

Status: harness and frozen dataset implemented on the private experiment
branch. TP4 A/B evidence is pending. This experiment cannot authorize a
default, YAML, `main`, or repository visibility change.

## Purpose

M1-132 showed full-model distribution drift for the M1-109 fused prefill
candidate, while M1-134 showed an exactly repeatable control path. Cross-arm
token or output differences are therefore real diagnostics, but are not by
themselves evidence of task-quality regression.

M1-137 supplies a predeclared, paired task-capability test for instruction
following:

- one-sided confidence: 95%;
- noninferiority margin: 2 percentage points;
- paired prompt count: 149;
- bootstrap samples: 20,000;
- bootstrap seeds: 20260729 for strict and 20260730 for loose;
- strict and loose Google IFEval prompt outcomes must both pass.

The sample count and margin were fixed before candidate responses were
observed. The 149-pair floor is the one-sided zero-regression bound for a 2%
margin. Paired bootstrap uses the prompt, not individual instructions or
generated tokens, as the resampling unit.

## Frozen data

- dataset: `google/IFEval`;
- organization: Google Research;
- license: Apache-2.0;
- dataset revision:
  `966cd89545d6b6acfd7638bc708b98261ca58e84`;
- evaluator revision:
  `e6890f85757dd84e27ca6df2dd30651dafad28e0`;
- source SHA-256:
  `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2`;
- frozen subset:
  `quality/external/google_ifeval/subset.power149.v2.jsonl`;
- subset SHA-256:
  `14dee74f7fc65768d326140367b31b57cce24d59e76bd0098b94d2730eef22e2`;
- manifest:
  `quality/external/google_ifeval/manifest.power149.v2.json`;
- manifest SHA-256:
  `01c7e9dd4aafc11b5e2505fec2c3c71c53d8d27992ab40445638e97404440107`;
- deterministic seed:
  `bi100-ifeval-power149-v2-seed-20260730`;
- coverage: all 25 instruction IDs, each represented at least 10 times.

The snapshot reuses the already pinned source, evaluator, license, wheels, and
request conversion from the 64-request M1-122 screen. It does not contain
competition requests, credentials, private traces, or model outputs.

Every service report is checked against the repository copy of the approved
manifest. The manifest filename, manifest SHA-256, subset SHA-256, and all 149
selected keys in canonical request order must match exactly. The offline
environment attestation must also list exactly the pinned wheel/archive
digests and the pinned `punkt_tab` archive digest. A changed key order,
substitute subset, missing wheel, extra wheel, or different NLTK resource
fails closed.

## Decision boundary

M1-137 has two qualification stages. The pre-cleanup aggregate can establish
that the paired 149-request statistics passed, but it explicitly leaves
`outer_lifecycle_pending=true` and cannot authorize the capability surface.
Only the final qualifier, run after scoped cleanup, can authorize the
two-point IFEval instruction-following capability surface. It requires:

- both arm runners and the paired noninferiority comparator to return zero;
- the aggregate to be reproduced from the retained arm reports, runtime
  contracts, offline-install attestations, 4xx reports, process identities,
  and comparison reports rather than trusted as a self-assertion;
- the pinned v2 layered-contract SHA-256 and recorded-session recovery to
  remain hash-bound;
- no emergency recovery of a live experiment process;
- three consecutive clean postflight observations;
- a successful four-card deterministic preflight;
- empty fatal and timeout scans, followed by an independent final rescan of
  the current log and return-code sets. The final report records only the
  input counts and set digests, never matching log lines.

Even that final authorization does not waive or replace:

- same-activation operator finite/error bounds;
- same-arm repeat and cache cold/warm exactness;
- API, SSE, tool, reasoning, structured-output, and multimodal contracts;
- code/math and long-context recall evidence;
- TP4 performance, lifecycle, fatal, timeout, Gloo, worker, and GPU gates.

Cross-arm exact-output comparison is retained as a diagnostic and never used
as the M1-137 task score. Semantic task evidence cannot override a failed
operator shadow.

## Runner

After building an offline IFEval environment from the power149 manifest and
installing an immutable runtime overlay from the exact clean source revision:

```bash
BI100_RUNTIME_SITE_PACKAGES=/path/to/runtime/site-packages \
BI100_RUNTIME_INSTALL_REPORT=/path/to/runtime/install.json \
scripts/run_m1_137_ifeval_power_ab.sh \
  INSTANCE /path/to/ifeval-env /tmp/m1-137-power149-SOURCE
```

The fixed arm order is fused-off then fused-on. Each arm uses a fresh TP4
service, `max_model_len=262144`, admission64/hybrid64, the submission kernel
profile, unchanged model/tokenizer/request semantics, scoped TERM cleanup,
recorded-session recovery, postflight, four-card preflight, and fatal/timeout
scans. `aggregate.json` contains only pre-cleanup statistical evidence;
`final_qualification.json` binds the completed lifecycle evidence and remains
incapable of authorizing performance, a default/YAML change, a `main` merge,
or production promotion.
