# M1-49 Zombie Cleanup False Failure (2026-07-24)

## Impact

The first TP4 M1-49 A/B stopped after a qualified `legacy40` pressure run and
never started `full_attention`. `overall.rc=1` and `cleanup.rc=1` were harness
failures, not model, GPU, or cache failures.

## Evidence

The service process group was `5307`. Worker PID `5659` exited early and was
adopted by PID 1 as `STAT=Z`. The service later logged normal FastAPI and engine
shutdown, and no CUDA, OOM, Gloo, SIGSEGV, worker-death, or Python fatal pattern
was present. The old cleanup implementation still used `pgrep -g 5307`, which
matches zombies, and reported:

```text
service process group 5307 survived cleanup
M1-49 service cleanup failed
```

A zombie cannot hold the API port or GPU memory and cannot be reaped by the
benchmark shell after PID 1 adopts it. Treating it as a live service caused a
false failure and repeated the cleanup wait in the EXIT trap.

## Correction

Commit `a24023c` introduced shared process-group cleanup which distinguishes
live states from `Z*`, retains TERM/KILL for live members, fails closed when
the process table cannot be inspected, and records zombie-only groups.

Commit `c47186d` hardened the fallback further:

- the `setsid` leader PID is retained as the expected PGID before `ps`
  verification;
- a missing verified PGID cannot enter an unbounded `wait`;
- a live leader must still belong to the recorded PGID;
- the A/B EXIT path also verifies that port 8000 is free;
- TERM-ignoring processes, process-table failure, and leader mismatch have
  dedicated regression tests.

## Validation

Local regression passed 415 tests with 25 dependency skips, and submission
preflight passed all eight checks. On the affected host, the new helper
classified PGID `5307` as `live=0, zombie=1`.

The fixed rerun completed both arms. Each cleanup reported one zombie-only
member, then proceeded to a free-port check and four-GPU preflight. Final
cleanup, comparison, and overall return codes were zero. The rerun therefore
demonstrates that the correction removes the false failure without allowing a
live process group to pass.

## Residual Risk

PID 1 in this environment does not promptly reap adopted zombies, so diagnostic
zombie rows can remain visible after a successful run. Escaped processes that
leave the original PGID are not inferred to be safe: port checks and subsequent
four-GPU preflights remain mandatory. A cleanup inspection error or live
member still fails the run.
