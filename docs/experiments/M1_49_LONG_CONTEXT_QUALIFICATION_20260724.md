# M1-49 TP4 Long-Context Qualification (2026-07-24)

## Scope

M1-49 `full_attention/admission64/direct` passed the fixed TP4 long-context
correctness and capacity gates. The qualified recovery source was `c06f63f`,
and its prerequisite capacity A/B was the qualified `a24023c` run. The service
used 10 physical full-attention KV layers, 67512 GPU blocks, 26214 CPU blocks,
and a 262144 maximum model length.

This qualification is deliberately scoped as
`hybrid-kv-capacity-correctness-not-prefill-speed`. Request elapsed times below
include generation and are not official TTFT, Input TPS, Cache TPS, or weighted
score measurements.

## Results

| Gate | Request | Prompt | Cached | Completion | Elapsed | Equivalence |
|---|---|---:|---:|---:|---:|---|
| 131K exact | cold | 131000 | 0 | 256 | 261.115s | same hash |
| 131K exact | warm | 131000 | 130992 | 256 | 41.573s | same hash |
| 235K warm-repeat | cold | 235000 | 0 | 1000 | 781.124s | reference only |
| 235K warm-repeat | warm 1 | 235000 | 234992 | 1000 | 221.894s | same warm hash |
| 235K warm-repeat | warm 2 | 235000 | 234992 | 1000 | 221.592s | same warm hash |
| 262K exact | cold | 262000 | 0 | 16 | 742.750s | same hash |
| 262K exact | warm | 262000 | 261984 | 16 | 10.372s | same hash |

All requests reached the fixed minimum completion count and reported
`finish_reason=length`. The 131K cold/warm pair, all three 235K requests, and
the 262K cold/warm pair each had identical privacy-safe message identities for
their required equivalence mode.

The quick API smoke suite passed 8/8. Multimodal prefix isolation passed all
five checks, including same-image reuse, different-image isolation, and exact
cold/warm output. Four-GPU preflights before and after the recovery passed and
matched. Fatal scan, process-group cleanup, final qualification, and overall
RC were zero; final `qualified=true` with no reasons.

## Decision

M1-49 is admitted to the selected 13-turn replay and the M1-48 235K prefill
profile. It is not yet a submission default and does not establish the final
TTFT, cache-hit, Output TPS, success-rate, or weighted-score thresholds.
`computility-run.yaml` remains unchanged.

Structured evidence:
`docs/experiments/evidence/M1_49_LONG_CONTEXT_QUALIFICATION_20260724.json`.

