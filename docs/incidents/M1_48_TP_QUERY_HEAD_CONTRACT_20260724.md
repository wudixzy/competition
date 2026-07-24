# M1-48 TP Query-Head Contract False Negative (2026-07-24)

## Incident

The M1-48 control and profile services completed the fixed 235000-token TP4
request successfully, but `profile_summary.rc` and `overall.rc` were one. All
other runtime, startup, GPU, fatal-scan, service, and cleanup gates returned
zero. The profile summary contained exactly 116 reasons:

```text
4 TP ranks * 29 prefill forwards = 116 paged-dispatch mismatches
```

The diagnostic expected eight query heads in every rank-local paged-attention
dispatch. The installed Qwen configuration has 16 global attention heads, so
tensor parallelism four produces exactly four query heads per rank. Runtime
counters consistently reported the correct value. This was a stale test
contract, not a model, GPU, or kernel failure.

## Correction

The profile summarizer now accepts the global attention-head count, requires it
to be positive and exactly divisible by TP size, and derives the rank-local
count. The fixed M1-48 harness passes 16 explicitly. Summary and qualification
artifacts bind both `num_attention_heads=16` and
`query_heads_per_rank=4`, preventing the same ambiguity from recurring.

Unit tests cover the real TP4 shape, a divisible alternate shape, and rejection
of invalid division. The qualifier independently recomputes the source log and
requires both head-count fields.

## Recovery

The original failed run remains immutable. The recovery script accepts only a
source run whose complete reason set is exactly the 116 expected stale-contract
messages and whose remaining gates all passed. It also rejects live source
process groups, a busy API port, missing evidence, or a failed fresh GPU
preflight before rebuilding the summary and qualification in a new directory.

The recovered report qualified with all return codes zero. It retains
`scope=post-m1-49-diagnostic-path-ranking-only` and
`promotion_authorized=false`; correcting the diagnostic does not authorize a
runtime, YAML, or `main` change.
