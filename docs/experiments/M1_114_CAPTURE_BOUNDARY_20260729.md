# M1-114 GDN Capture-Boundary Prefill Alignment

## Status

Targeted capture-boundary correctness passed. The combined M1-109 fused
prefill candidate did not qualify its full TP4 comparison.

This experiment is not authorized for `main`, `computility-run.yaml`, or a
production default.

## Trigger

M1-109 pair 2 control reproduced a 235K correctness failure with fused prefill
disabled:

- cold and both warm requests had the same first generated token;
- both warm requests were mutually deterministic;
- the cold full-output hash differed from the warm full-output hash;
- the warm path restored the 234992-token GDN state and physically replayed
  eight prompt tokens;
- the cold path computed those final eight tokens inside a 5624-token model
  forward.

No fatal, OOM, worker-loss, or GPU lifecycle failure accompanied the mismatch.
The M1-109 candidate therefore remains rejected even though its cold prefill
performance improved.

## Single Variable

For `admission64`, a scheduler prefill step now stops at the earliest pending
GDN capture boundary that lies strictly inside its physical query interval.
Already reached boundaries and boundaries outside the current step do not
change scheduling.

For the 235000-token case:

```text
old cold partition: 229376 -> 235000  (5624 query tokens)
new cold partition: 229376 -> 234992  (5616 query tokens)
                    234992 -> 235000  (8 query tokens)
warm partition:     restore 234992 -> 235000 (8 query tokens)
```

The saved state and the replay suffix are therefore produced by the same
physical model-forward partition on cold and warm paths. The helper validates
content keys and fails closed on invalid token geometry. It does not change
tokens, sampling, model math, precision, context capacity, or request
semantics.

`fine32` remains unchanged by this experiment.

## Qualification

1. Run the policy and scheduler unit tests in the CoreX runtime.
2. Build an immutable runtime overlay from the committed source and verify its
   identity.
3. Run `scripts/run_m1_99_fused_prefill_service_ab.sh`.
4. The first fused-off arm must show exact cold/warm output hashes at 32K, 65K,
   131K, and 235K. Any mismatch stops the run.
5. If the first arm passes, complete all three fused-off/fused-on TP4 pairs.
6. Require clean scoped cleanup, postflight, repeated four-GPU preflight, and
   no fatal, timeout, Gloo reset, or worker-loss signature.

The cold TTFT cost of the extra suffix forward must be reported separately from
M1-109 fused-prefill gains. Full quality gates remain required before any
promotion.

## TP4 Result

The fixed three-pair alternating run completed all six full-model TP4 arms on
`ssh-73ca29ba` from
`2014b7e2451e10a0a4c9a51cade794d81f996901`.

Within every individual arm, cold, warm 1, and warm 2 had identical first-token
and full-output hashes at 32K, 65K, 131K, and 235K. Warm cached-token counts
were respectively 32,752, 65,520, 131,056, and 234,992. The 235K path
therefore restored the aligned state and replayed exactly eight prompt tokens.
This passes the targeted M1-114 cache-boundary correctness condition.

The fused candidate produced reproducible cold-prefill improvements:

| Prompt | Pair improvements | Median |
|---|---|---:|
| 32K | 17.70%, 12.22%, 3.32% | 12.22% |
| 65K | 23.38%, 19.58%, 10.96% | 19.58% |
| 131K | 30.36%, 26.52%, 19.44% | 26.52% |
| 235K | 36.71%, 32.78%, 27.90% | 32.78% |

The fused-off and fused-on arms retained identical first generated tokens and
completion structure. Their full outputs diverged at 65K in all three pairs
and at 235K in pairs two and three. Pair three also had a 235K warm slowdown
of 0.590 seconds, exceeding the 0.5-second individual limit. The fused
candidate therefore failed the strict output gate despite its repeatable
prefill speedup.

All six arm measurements, fatal scan, timeout scan, source identity, and final
postflight passed. Recorded-session recovery found no live processes, but each
service group initially contained one PID-1-owned worker zombie. The zombies
then disappeared and the machine-wide postflight was clean; the stricter
no-recovery lifecycle qualifier correctly remained failed. M1-118 addresses
this harness-level descendant reaping gap separately.

The result authorizes focused capability adjudication only. It does not
authorize production promotion, `main`, YAML, or an official-style replay.

Structured evidence:
`docs/experiments/evidence/M1_114_CAPTURE_BOUNDARY_20260729/qualification.json`.
