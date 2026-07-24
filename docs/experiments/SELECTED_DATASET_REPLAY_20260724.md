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

```bash
python3 scripts/replay_selected_dataset.py \
  --dataset chat_dataset_v0.json \
  --label m1-49-selected-dataset \
  --max-tokens 256 \
  --out bench_runs/m1_49/selected_dataset/report.json
```

The report contains request dimensions, usage counters, TTFT, decode TPS,
cache counters, finish reasons, and SHA-256 output identities. It does not
contain raw prompts, images, assistant text, reasoning text, or tool calls.
`bench_runs/` is ignored by Git so local diagnostic artifacts cannot be added
accidentally.

## Interpretation

Use the report to detect request failures, chat-template regressions, unstable
outputs, and loss of session-prefix reuse. Its weighted value is explicitly a
small-sample proxy. Promotion still requires the frozen TP4 gates and the final
competition thresholds.
