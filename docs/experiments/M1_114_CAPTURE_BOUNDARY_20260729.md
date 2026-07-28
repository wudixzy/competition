# M1-114 GDN Capture-Boundary Prefill Alignment

## Status

Implementation complete; TP4 qualification pending.

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
