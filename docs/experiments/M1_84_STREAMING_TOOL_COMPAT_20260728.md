# M1-84 streaming tool compatibility

Date: 2026-07-28

## Objective

The local workload description says all 881 official requests use streaming.
The qualified M1-72 single-GPU A/B covered explicit `strict=false` and
object-form tool-call history only through non-streaming HTTP responses.
M1-84 closes that coverage gap before either request-compatibility change is
considered for a full-model TP4 candidate.

This is a diagnostic and correctness-gate change. It does not modify model
weights, runtime request semantics, cache behavior, sampling, chat templates,
`computility-run.yaml`, or any default optimization switch.

## Gate extension

`tests/qwen36_tool_http_gate.py` now adds four deterministic streaming cases:

1. canonical function tool definition with omitted `strict`;
2. the equivalent definition with explicit `strict=false`;
3. canonical JSON-string arguments in assistant tool-call history;
4. the equivalent object-form arguments in assistant tool-call history.

Successful requests must pass the shared quality-gate SSE parser. The parser
requires:

- `text/event-stream`;
- one final SSE frame boundary;
- exactly one `[DONE]`;
- exactly one terminal `finish_reason`;
- exactly one final usage chunk;
- valid chunk identity and schema;
- valid reconstructed content, reasoning, and tool calls;
- JSON object arguments for reconstructed tool calls;
- internally consistent prompt, completion, and total token counts.

Only privacy-safe summaries are retained. Generated content, reasoning, tool
schemas, tool arguments, raw SSE, and request bodies are represented only by
digests or aggregate fields.

For an accepting candidate, explicit `strict=false` must match the omitted
form, and object-form history must match JSON-string history. The comparison
includes reconstructed semantic output, terminal reason, prompt and completion
token counts, content/reasoning presence, and tool-call count. Cached-token
usage is validated and recorded but is not required to be equal because
sequential equivalent requests can legitimately observe different cache
warmth.

The existing malformed-arguments, unsupported `strict=true`, unsupported
`tool_choice=required`, and post-error health contracts remain unchanged.
The A/B comparator still requires exactly three expected candidate-side 4xx
responses with complete privacy-safe attribution.

## Local qualification

The unit gate covers:

- control behavior that reproduces both compatibility 400 responses in
  non-streaming and streaming modes;
- candidate behavior that accepts both forms with exact deterministic output;
- fail-closed rejection of non-streaming or streaming output drift;
- malformed streaming usage rejection;
- A/B cross-overlay output stability;
- diagnostic-runner result propagation;
- privacy constraints.

The branch remains local-ready until a healthy BI100 is available. No real
weight, HTTP, GPU, TP4, semantic-quality, long-context, throughput, or official
881-request result is claimed by the local tests.

## Remote acceptance sequence

1. Run one real-weight single-GPU control/candidate A/B using the fixed
   diagnostic checkpoint and exact runtime overlays.
2. Require all 13 HTTP cases, streaming equivalence, 4xx attribution,
   process-group cleanup, repeated GPU preflight, and fatal/timeout scans to
   pass.
3. Preserve scoped cleanup: SIGTERM, a 60-second grace period, SIGKILL only
   for verified survivors, then wait/reap and postflight.
4. Only after that result, run the full-model TP4 functional and performance
   A/B. M1-84 alone does not authorize a default change, YAML change, `main`
   merge, or production promotion.
