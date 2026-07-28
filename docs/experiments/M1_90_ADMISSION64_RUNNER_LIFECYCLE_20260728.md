# M1-90 admission64 runner lifecycle

Date: 2026-07-28

## Objective

M1-90 makes the pending M1-85 TP4 `fine32/direct` versus
`admission64/direct` quality A/B safe to interrupt and strict enough to reject
incomplete cleanup evidence. It does not change model execution, cache
implementation, request semantics, sampling, tokenizer, chat template,
`computility-run.yaml`, Dockerfile, or any default optimization switch.

The implementation is commit `9334866` on the private branch
`fix/M1-90-admission64-runner-lifecycle-20260728`, based on
`fix/M1-89-multimodal-cache-namespace-20260728@1d83a10`.

## Audit findings

The previous runner had four lifecycle gaps:

1. the outer interrupt path waited 900 seconds and identified its child only
   by PID and PGID;
2. the child wrote `fine32/admission64_child_identity.json`, while recovery
   expected `control/candidate_child_identity.json`;
3. the API service entered a nested session without starttime or token
   attestation, so a hard-killed child runner could leave an undiscoverable
   service;
4. timeout and fatal scans covered only a narrow artifact subset.

The filename mismatch alone would have made a completed run fail its recovery
qualification. PID or PGID without starttime and a session token also cannot
safely distinguish a live experiment process from later PID reuse.

## Implementation

Both A/B children and both nested API services now start through
`scripts/exec_bi100_session.py`. Before `exec`, it atomically writes a private
session-v1 identity containing:

- `pid=pgid=sid`;
- Linux `/proc` starttime ticks;
- a random 32-hex session token inherited by that process tree.

Normal cleanup validates all fields, sends SIGTERM to only that attested
process group, waits 60 seconds, sends SIGKILL only to verified survivors,
then waits/reaps. PID-only fallback also requires an unchanged starttime.
Finalizers ignore repeated TERM/INT while bounded cleanup is running.

The inner quality runner performs its own recorded-session recovery and
postflight. The outer runner additionally scans four identities in fixed
order: control runner, control service, candidate runner, candidate service.
This outer scan handles the case where a child runner was killed before its
`finally` block could clean the nested service.

`tests/qualify_recorded_session_cleanup.py` separates cleanup from evidence
qualification. Emergency TERM or KILL may restore the machine, but the
experiment qualifies only when every expected identity was already quiescent,
the full token scan completed, no process escaped its group, and no signal was
needed. Malformed, missing, reordered, duplicated, or incomplete evidence
fails closed.

Quality-service status and the admission64 aggregate are now v2. Their SHA-256
bindings include process identities and inner recovery reports. The aggregate
also validates process identity structure and rejects reused A/B session
tokens. Fatal scans include CUDA/CoreX errors, OOM, device assertions, Gloo,
NCCL, worker loss, connection reset, watchdog/runner timeouts, missing GDN
state, and non-finite GDN output. Every numeric `*.rc` is inspected for
timeout/forced-termination values `124`, `137`, and `143`.

## Local validation

No BI100 GPU or remote service was used for this result.

- focused lifecycle and fail-closed tests: `37/37` passed;
- complete test discovery: `1050` passed, `25` environment-dependent skips;
- submission preflight: `9/9` passed;
- quality-data manifest: qualified, 12 long-context and 11 Agent cases;
- official metric manifest: qualified, 53 cases;
- both shell runners pass `bash -n`;
- Python compilation and Git whitespace checks pass.

These results validate runner logic and evidence contracts only. They do not
establish model quality, cold/warm consistency, TP4 stability, cache benefit,
TTFT, throughput, or an official score.

## Next run

After a healthy four-card host and a source-bound immutable overlay are
available:

```bash
export BI100_RUNTIME_SITE_PACKAGES=/absolute/immutable/site-packages
scripts/run_m1_85_admission64_quality_ab.sh \
  private-tp4-instance \
  /tmp/m1-85-admission64-quality-ab-YYYYMMDDTHHMMSSZ
```

Only a fully qualified v2 result may authorize the functional and Agent
non-regression dimension. It still cannot authorize `main`, YAML, a default
policy change, or a performance claim without the separate TP4 performance,
cache-correctness, long-context, and stability gates.
