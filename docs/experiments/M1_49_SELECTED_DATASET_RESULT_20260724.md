# M1-49 Selected Dataset Result (2026-07-24)

## Scope

The frozen selected dataset was replayed against a fresh TP4
`full_attention/admission64/direct` service at source `7324b89`. The qualified
M1-49 long-context result was a hard prerequisite. Dataset SHA, four-session
shape, 13-turn order, seed, 256 maximum output tokens, concurrency one, runtime
contract, process cleanup, and before/after GPU preflights were fixed.

This is a supplemental 13-turn result. It is not the complete 881-request
trace, does not reproduce the evaluator's traffic mix, and cannot be compared
directly with the official weighted-score threshold of 8000.

## Results

| Metric | Result | Competition threshold on official trace |
|---|---:|---:|
| Request success | 100% (13/13) | >=99% |
| Output TPS P10 | 21.309 | >=20 |
| TTFT P90 | 3.089s | <=5s |
| Aggregate cache hit | 50.19% | >=50% |
| Prompt tokens | 6,089 | n/a |
| Cached prompt tokens | 3,056 | n/a |
| Uncached prompt tokens | 3,033 | n/a |
| Completion tokens | 3,216 | n/a |
| Replay wall time | 181.117s | n/a |

All four headline thresholds that can be observed on this small trace passed.
Output decode rates ranged from approximately 21.29 to 22.59 TPS. The first
conversation's cold TTFT was 4.509s; subsequent selected turns were between
approximately 1.67s and 3.11s. The cache-hit result is only 0.19 percentage
points above the required threshold, so it provides little safety margin.

The replay's diagnostic formula produced Input TPS residual proxy 86.995,
Cache TPS proxy 87.655, and weighted proxy 650.493. These denominators reflect
only 6,089 prompt tokens and sequential per-request TTFT on this selected
sample. The value must not be compared with 8000 or extrapolated to an official
score.

## Integrity

Startup, startup-to-M1-49 matching, replay, qualification, fatal scan, cleanup,
both GPU preflights, preflight comparison, and overall RC were zero. The
qualification was true with no reasons and scope
`selected-13-turn-supplemental-not-official-score`.

The committed evidence contains no prompt, image, assistant text, reasoning,
or tool-call payload. The runtime report retained only token counts, timings,
finish reasons, and SHA-256 output identities.

## Decision

The selected replay passes as supplemental evidence and does not block M1-48
profiling. It does not promote M1-49 to the submission default. Promotion still
requires the full evaluation-like replay and final score thresholds.

Structured evidence:
`docs/experiments/evidence/M1_49_SELECTED_DATASET_RESULT_20260724.json`.

