# M1-118 session subreaper

## Purpose

M1-118 makes every long-running service session a Linux child subreaper so
that exited TP4 workers are reaped by the experiment process instead of being
left to PID 1. Cleanup remains scoped by PID, PGID, SID, `/proc` starttime, and
a private per-session token.

## First TP4 qualification

The first real M1-116 control arm used source `c0bac0c` and runtime overlay:

- source: `/root/m1-118-source-c0bac0c-exact`
- overlay: `/root/m1-118-runtime-c0bac0c/site-packages`
- run: `/tmp/m1-116-fused-quality-c0bac0c-20260729-v1`
- instance: `ssh-73ca29ba`

The service completed its output diagnostic, 53-case quality suite, and
11-case Agent suite, but cleanup stopped before the candidate arm. The inner
session leader had the inherited outer token in `/proc/<pid>/environ`, while
its children had the newly assigned inner token. Linux exposes the process's
initial environment through this interface; mutating `os.environ` after
startup did not update the leader identity observed by the recovery tool.

The recovery tool correctly failed closed and did not signal the mixed-token
group. The run is therefore invalid as an A/B:

- outer return code: `1`
- candidate arm: not started
- service recovery qualification: failed
- orchestrator postflight and per-GPU preflight after scoped recovery: passed
- production promotion: not authorized

The control workload results remain diagnostic evidence only:

- quality suite: 52 passed, 1 failed
- Agent workload: 10 passed, 1 failed
- failed cases: `max_tokens_1` and `stream_forced_terminal`
- fatal GPU/runtime errors: none
- request-validation diagnostics appeared in the active runtime log with
  safe `validation_field` and `validation_type` summaries

After verifying both nested identities, every current PID starttime, SID,
PGID, and token assignment, the stuck inner service group received SIGTERM.
It became quiescent in about 11 seconds; SIGKILL was not used.

## Fix

Commit `2a752261ea53d3ade9ef9a5fcfad0ad203f7d148` creates the private session,
generates its token, and then re-executes the same PID with that token in the
initial environment before writing the identity file. The internal re-exec
marker is removed before child launch so a nested helper creates a distinct
session and token.

The recovery checks were not weakened. A token, starttime, session, or process
group mismatch still fails closed.

## Verification

Local verification:

- 20 focused lifecycle and recovery tests passed
- full suite: 1211 passed, 26 skipped
- submission preflight: 9 of 9 checks passed
- `git diff --check`: passed

Remote exact-source verification:

- source: `/root/m1-118-source-2a75226-exact`
- 20 focused tests passed
- submission preflight: 9 of 9 checks passed
- nested outer/inner session recovery qualified
- inner cleanup used SIGTERM and no SIGKILL
- both recorded identities were quiescent after the runner exited

This is a lifecycle qualification only. A fresh full TP4 M1-116 A/B is still
required before any performance or quality conclusion.

