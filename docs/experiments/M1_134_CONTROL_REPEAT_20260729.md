# M1-134 teacher-forced control repeat

Date: 2026-07-29

Status: valid TP4 control/control calibration. All 320 sampled positions were
identical. This result calibrates measurement repeatability but does not
authorize promotion.

## Purpose

M1-132 observed large teacher-forced differences between fused-off and
fused-on services. M1-134 measures fresh-service control repeatability before
assigning those differences to M1-109. It is an attribution experiment, not a
candidate promotion or performance experiment.

Both arms run with `BI100_ATTN_COREX_FUSED_PREFILL=0`. The runner still checks
one source revision, one immutable runtime tree, TP4, 262144 capacity, fixed
request identity, cold-only observations, lifecycle health, and the same
predeclared teacher-forced thresholds. The comparator explicitly requires
control-mode observations from both arms, so an accidentally enabled fused
path makes the report invalid.

Private arm observations are removed from the outer `trap` on success,
failure, timeout, or interruption. The retained comparison contains aggregate
counts and deltas only.

## Command

```bash
BI100_RUNTIME_SITE_PACKAGES=/path/to/site-packages \
BI100_RUNTIME_INSTALL_REPORT=/path/to/install.json \
scripts/run_m1_134_teacher_forced_control_repeat.sh \
  INSTANCE /tmp/m1-134-control-repeat-SOURCE
```

M1-134 cannot authorize a default selector, `computility-run.yaml`, `main`,
repository visibility, or production promotion.

## Result

Both fresh fused-off arms completed on `ssh-73ca29ba` from source
`00c83bc86ca6aa95f5d9b0e01cd5327cbd3b2341`, using the same immutable runtime
tree as M1-132. All arm, comparison, cleanup, recovery, postflight, GPU
preflight, fatal-scan, and timeout-scan return codes were zero.

| Metric | Result |
|---|---:|
| Prompt cases | 5 |
| Sampled positions | 320 |
| Top-1 agreement | 1.0 |
| Top-1 mismatches | 0 |
| Teacher-token logprob max delta | 0.0 |
| Teacher-token logprob p99 delta | 0.0 |
| Shared top-k logprob p99 delta | 0.0 |
| Mean teacher-token NLL delta | 0.0 |

Private arm observations were absent after the runner completed. The shell
used to launch the detached runner remained as an unreaped zombie after all
runner outputs had been atomically written; this is an external launcher
reaping issue, not a service, worker, cleanup, postflight, or GPU failure.

The exact A/A result makes the M1-132 fused-off/fused-on distribution drift
attributable to the candidate path. Under the v2 layered gate, attribution
does not turn a full-model distribution observation into an operator numeric
failure: M1-109 must now pass a same-real-activation shadow reference and a
powered paired capability noninferiority evaluation.

Privacy-safe evidence is in
`docs/experiments/evidence/M1_134_CONTROL_REPEAT_20260729/summary.json`.
