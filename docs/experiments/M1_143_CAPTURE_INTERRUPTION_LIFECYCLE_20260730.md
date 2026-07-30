# M1-143 capture interruption lifecycle

Date: 2026-07-30

Status: invalid interrupted run; lifecycle defect fixed on the private
experiment branch. This run provides no operator, TP4 performance, capability,
or promotion evidence. Formal YAML and `main` remain unchanged.

## Observed run

The qualification capture used source
`076c82d990d234f491a27d2dad6bccf43f25839d`, instance
`ssh-73ca29ba`, one TP4 service startup, and the frozen 32K/65K/131K capture
matrix. It was deliberately stopped during `service_startup` in response to an
interactive cleanup request.

Only privacy-safe runner artifacts were inspected:

- the startup stage ran for 524.342 seconds before interruption;
- scoped cleanup found six live processes in the recorded service session;
- cleanup sent `SIGTERM`, observed the full 60-second grace period, then sent
  `SIGKILL`;
- cleanup ended with zero live processes and three PID-1-owned zombies;
- the GPU process table was empty after cleanup and each card showed only the
  257 MiB driver baseline;
- no activation bank, prompt, token, response, or model output was retained;
- the fatal scan observed two worker-loss lines after interruption, so the run
  correctly remained unqualified.

The zombies cannot hold GPU resources and cannot be reaped by the experiment
runner after they have been adopted by PID 1.

## Runner defect

The interrupted run did not create `postflight_after.json`. The capture runner
placed fatal scanning, postflight, repeated preflight, preflight comparison,
and source verification in one `try` block. A fatal-scan failure therefore
skipped every later lifecycle check.

The runner now executes terminal checks independently:

1. scoped TERM-first cleanup;
2. final process/GPU postflight;
3. repeated four-card preflight and comparison when postflight is clean;
4. fatal-category scan;
5. source-tree identity check.

Every attempted stage records its own gate. A failed postflight explicitly
marks repeated preflight and comparison as not run, while fatal and source
checks still execute. The first failed stage is retained as the terminal stage
instead of being overwritten by a later secondary failure.

## Validation

Targeted lifecycle tests cover both paths that exposed the defect:

- a startup failure plus fatal worker-loss evidence still runs final postflight,
  repeated preflight, comparison, fatal scan, and source verification;
- a failed final postflight still runs fatal and source verification while
  preventing an unsafe repeated preflight.

The next capture must use a clean exact-commit worktree containing this fix.
It must produce `postflight_after.json`, `preflight_after.json`, and
`preflight_comparison.json` before its activation bank can authorize replay.
