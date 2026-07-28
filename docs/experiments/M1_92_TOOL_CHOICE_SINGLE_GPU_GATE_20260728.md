# M1-92 tool-choice single-GPU gate

Date: 2026-07-28

Branch: `exp/M1-92-tool-choice-gate-20260728`

## Objective

The historical unbound platform result recorded 226 tool-request 4xx
responses among 690 tool-bearing requests. M1-68 and M1-84 cover explicit
`strict=false`, assistant tool-call history represented as either a JSON
string or an object, streaming SSE reconstruction, and attributed malformed
request failures. They did not exercise the valid tool-choice modes used by
the frozen quality contracts:

- tools present with `tool_choice` omitted, which must default to `auto`;
- explicit `tool_choice="auto"`;
- a named function choice object.

M1-92 adds a diagnostic service gate for those three modes in both
non-streaming and streaming requests. It changes no serving code, model
weights, dtype, tokenizer, chat template, cache behavior, sampling semantics,
formal YAML, Dockerfile, or production default.

## Gate

`tests/qwen36_tool_choice_http_gate.py` sends six fixed greedy requests using
temperature zero and seed `20260728`. Every request must:

- return HTTP 200;
- produce exactly one structurally valid function call;
- finish with `tool_calls`;
- use the requested function;
- produce a JSON-object argument with the requested city;
- return valid usage accounting;
- satisfy the existing strict SSE framing, final usage, and `[DONE]`
  contracts when streaming.

Omitted and explicit `auto` are equivalent request semantics, so their
privacy-safe semantic digests must match within each transport. Each of the
three modes must also match between non-streaming and streaming transports.
Named selection is not required to match `auto`, because they are distinct
request semantics. Tool-call IDs are deliberately excluded from equality.

The report retains only digests, counts, booleans, timings, and model path. It
does not retain the prompt, function name, arguments, response text,
reasoning, raw SSE, request body, or credentials.

## Unsupported modes

The current CoreX vLLM 0.6.3 baseline rejects `tool_choice="required"` and
tool schemas with `strict=true`. Those are compatibility gaps, not malformed
OpenAI requests and not successful-request evidence. M1-92 does not claim to
implement or validate them. They remain explicitly marked unevaluated rather
than being converted to synthetic HTTP 200 responses.

The fixed official 53-case manifest requires `auto` and named function
selection. It does not justify changing production behavior for
`required`/`strict=true` without a separate implementation and model-capability
gate.

## Queue integration

The existing M1-84 TP1 diagnostic service now runs M1-92 after its M1-84
streaming tool-history gate. Its status binds the complete M1-92 report SHA
and summarizes:

- all six valid requests returned HTTP 200;
- omitted and explicit `auto` were exact;
- each mode was exact across non-streaming and streaming;
- all tool calls were structurally valid;
- `required` and `strict=true` were not evaluated.

M1-87 is upgraded to the v4 aggregate contract and rejects a missing, extra,
tampered, malformed, or failed M1-92 gate or artifact. Existing attested
process-group cleanup, TERM 60-second grace, survivor-only KILL, wait/reap,
postflight, repeated GPU preflight, fatal scan, and timeout scan remain
unchanged.

## Current result

Local validation at the uncommitted branch tip:

- M1-92/M1-87 focused contracts: 37 passed;
- complete tests-root discovery: 1079 passed, 25 dependency skips;
- submission preflight: 9 of 9 passed;
- fixed quality-data manifests: 12 long-context and 11 Agent cases passed;
- official metric manifest: 53 cases passed;
- Python and shell syntax checks passed.

The current host has no CoreX GPU, and the latest bounded probe of
`ssh-73ca29ba` still failed in the TLS ProxyCommand layer before SSH
authentication. No diagnostic model, remote process, or GPU command ran.

Therefore M1-92 has no real HTTP, model, single-GPU, full-model, TP4,
semantic-quality, throughput, or official-score result yet. A passing
four-layer diagnostic run would qualify only request plumbing and structural
tool output. The complete full-model functional, long-context, cold/warm,
quality, stability, and performance gates remain mandatory.
