# M1-167 Chat Completions field interactions

## Scope

This follow-up audits interactions between supported Chat Completions fields
after M1-165 added `max_completion_tokens`. It keeps Pydantic
`extra="forbid"`, does not accept unsupported lookalike fields, and does not
change model, tokenizer, chat-template, sampling, YAML, or cache defaults.

The fixed contract covers eight relationships:

- completion token budget precedence;
- top-level and nested thinking precedence;
- `stream` and `stream_options` dependency;
- `logprobs` and `top_logprobs` dependency;
- tools and `tool_choice` validation;
- mutually exclusive structured-output sources;
- nested `response_format.json_schema` shape;
- `continue_final_message` and generation-prompt compatibility.

Malformed multi-field payloads must fail as HTTP 400 and must not become a
server exception. Deprecated or endpoint-specific lookalikes `function_call`,
`functions`, `max_output_tokens`, and `reasoning_effort` remain
`extra_forbidden` until a separately specified lossless mapping exists.

## Runtime result

The repository protocol was copied into an immutable copy of the qualified
M1-165 CoreX runtime overlay on `ssh-73ca29ba`. The probe imported the actual
CoreX vLLM 0.6.3 `api_server` and `protocol` modules and exercised their ASGI
request-validation and sampling-conversion path without loading a model.

- 18/18 interaction cases matched;
- accepted cases entered serving and exposed the expected sampling state;
- rejected cases stopped before serving with HTTP 400;
- zero HTTP 500 responses;
- runtime `protocol.py` SHA-256 matched the audited local source;
- no GPU or model-quality claim is made by this protocol-only probe.

Evidence is stored in
`docs/experiments/evidence/M1_167_CHAT_FIELD_INTERACTIONS_20260731`.

## Verification

```bash
python3 -m unittest \
  tests.test_protocol_unit \
  tests.test_chat_request_compat_field_audit_unit \
  tests.test_chat_field_interactions_runtime_probe_unit \
  tests.test_m1_165_max_completion_tokens_evidence_unit \
  tests.test_api_4xx_telemetry_unit \
  tests.test_summarize_api_4xx_log_unit
python3 tests/audit_chat_request_compat_fields.py --root .
python3 tests/submission_preflight.py --root .
```

The focused suite passed 71 tests and submission preflight passed all checks.
The formal `computility-run.yaml` is unchanged.
