# Selected Dataset Replay Contract (2026-07-24)

## Scope

`chat_dataset_v0.json` is the selected local regression dataset. Its frozen
identity and shape are:

- SHA-256: `dac6afc77621b51dbc09cfa046c008a1e51a779bb771edcb27cb6a686f8884c8`
- 4 conversations
- 13 sequential user turns

This dataset exercises multi-turn chat-template handling and same-session
prefix reuse. It is supplemental evidence only. It is not the complete
881-request evaluation trace and cannot qualify a cache policy, estimate the
official weighted score, or replace the fixed long-context gates.

## Execution Order

Do not inject these requests into an active fixed A/B run. Run the replay only
after the candidate has passed its capacity A/B and 131K/235K/262K gates, with
a freshly restarted service and an empty cache. Control and candidate runs must
use the same dataset identity, request order, seed, and `max_tokens`.

Use the fixed service-lifecycle harness after providing the qualified M1-49
long-context evidence and the atomic runtime overlay:

```bash
BI100_RUNTIME_SITE_PACKAGES=/path/to/runtime_overlay/site-packages \
M1_49_LONG_DIR=bench_runs/m1_49/full_attention_long \
bash scripts/run_m1_49_selected_dataset.sh
```

The harness rejects dataset SHA or shape drift before starting the service. It
uses a fresh `full_attention/admission64/direct` TP4 service, makes the selected
dataset replay its first API workload, performs before/after four-GPU
preflights, and uses process-group cleanup. The replay command is frozen at 256
maximum output tokens, seed 20260713, and sequential concurrency one.

The report contains request dimensions, usage counters, TTFT, decode TPS,
cache counters, finish reasons, and SHA-256 output identities. It does not
contain raw prompts, images, assistant text, reasoning text, or tool calls.
`bench_runs/` is ignored by Git so local diagnostic artifacts cannot be added
accidentally.

`tests/qualify_selected_dataset_replay.py` recomputes all aggregate metrics
from the 13 redacted turns and rejects failures, non-finite values, cache
accounting errors, metric tampering, and identity drift. Its qualification
scope is `selected-13-turn-supplemental-not-official-score`; a zero RC confirms
evidence integrity only.

## Interpretation

Use the report to detect request failures, chat-template regressions, unstable
outputs, and loss of session-prefix reuse. Its weighted value is explicitly a
small-sample proxy. Promotion still requires the frozen TP4 gates and the final
competition thresholds.
