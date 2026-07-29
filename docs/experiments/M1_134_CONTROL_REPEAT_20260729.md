# M1-134 teacher-forced control repeat

Date: 2026-07-29

Status: implemented; TP4 execution pending.

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
