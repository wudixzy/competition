# E-SYNC-01 GDN Finite-Check Mode - 2026-07-13

## Hypothesis

The model applied two host-visible `torch.isfinite(...).all()` reductions in
each of 30 GatedDeltaNet layers for every decode token. Making these checks
qualification-only should reduce inter-token latency without changing model
math or the fixed evaluator contract.

## Candidate

```text
baseline  5e06675c9a9f07dc513c17b7eb1cef2a7c0d35fd
candidate 916acd6  perf(gdn): make per-layer finite checks opt-in
```

`BI100_GDN_FINITE_CHECK=0` is the performance default. Setting it to `1`
restores the synchronous per-layer checks. The diagnostic-only
`BI100_GDN_ALLOW_NAN_ZERO=1` setting also forces finite checking on so it cannot
silently become a no-op.

No model weights, scheduler behavior, attention math, cache semantics, or
`computility-run.yaml` arguments changed.

## Method

The baseline and candidate used the same four-GPU fixed launch command. Each
side ran three sequential groups with:

```text
requests=3
workers=1
max_tokens=128
prompt_repeat=126
seed=20260713
prompt_salt=E-SYNC-01-{a,b,c}
```

Artifacts are under:

```text
bench_runs/20260713_E_SYNC_01/baseline
bench_runs/20260713_E_SYNC_01/candidate
```

## Performance

| Metric | Baseline runs | Candidate runs | Median change |
| --- | --- | --- | ---: |
| Decode TPS P10 | 8.012 / 8.050 / 8.056 | 8.330 / 8.410 / 8.479 | +4.46% |
| ITL P50 (s) | 0.12468 / 0.12402 / 0.12455 | 0.11638 / 0.11614 / 0.11583 | -6.75% |
| ITL P90 (s) | 0.12770 / 0.12732 / 0.12675 | 0.12855 / 0.12693 / 0.12678 | -0.31% |
| TTFT P90 (s) | 4.458 / 4.292 / 4.338 | 6.341 / 4.390 / 4.585 | +5.70% |
| Overlap score | 637.39 / 642.88 / 590.95 | 629.40 / 670.14 / 617.04 | -1.25% |
| Disjoint score | 343.21 / 345.87 / 324.38 | 343.03 / 360.72 / 339.37 | -0.05% |

All groups completed with a 100% request success rate. Candidate B and C
improved both score formulas; A had a cold TTFT outlier. This experiment is a
decode-path optimization, not a demonstrated TTFT or aggregate-score winner.

The direct decode indicator, ITL P50, improved by 6.75% and therefore exceeds
the plan's 5% E-SYNC-01 keep threshold. Decode TPS P10 improved in every pair.

## Correctness And Stability

- all three fixed-prompt messages: byte-identical;
- finish reason, prompt-token count, and completion-token count: identical;
- full API smoke: 14/14 pass;
- non-GPU unit tests: 65/65 pass;
- focused finite-mode plus P0 tests: 40/40 pass;
- qualification import with `BI100_GDN_FINITE_CHECK=1`: pass;
- `BI100_GDN_ALLOW_NAN_ZERO=1` forces checking: pass;
- 99,500-token cold/warm gate: pass;
- long-context cold/warm message SHA-256: identical;
- cached tokens on warm 99.5K request: 99,296;
- non-finite, OOM, CUDA, NCCL, engine-dead, or fatal errors: zero;
- fixed evaluator YAML SHA-256: unchanged.

The 99.5K run took 152.629 seconds cold and 19.111 seconds warm. These values
are qualification evidence only and are not used as a paired performance claim.

## Decision

**Keep as a decode winner.** The candidate removes synchronous diagnostics from
the formal hot path, preserves an explicit qualification mode, passes every
correctness gate, and exceeds the planned P50 decode threshold. Do not attribute
TTFT or aggregate-score improvement to this change; later experiments must use
the candidate as the new decode baseline and continue targeting the large gap
to the Output TPS 20 hard requirement.

Rollback is a single revert of the candidate commit or setting
`BI100_GDN_FINITE_CHECK=1` for qualification/debug runs.
