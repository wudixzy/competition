# Platform HTTP 4xx attribution

Date: 2026-07-27

## Observation

The platform log contains adjacent Chat Completions responses at
2026-07-26T16:59:30:

```text
POST /v1/chat/completions HTTP/1.1 400 Bad Request
POST /v1/chat/completions HTTP/1.1 200 OK
```

An access-log status alone cannot establish whether the 400 is an expected
request-validation case or a model compatibility failure. The official
functional surface intentionally sends invalid bodies, missing message fields,
empty messages, and invalid parameter bounds. Those requests must return 4xx,
not 200 or 5xx.

The fixed submission contract also has `max_num_seqs=1`. The inherited vLLM
0.6.3 scheduler can deadlock its waiting queue when a single request asks for
`n=2`, so the current direct-engine guard returns a normalized 400. The
historical platform result reported `t2_n_2=0`, making this the leading
explanation for one isolated 400 during the functional phase. It is not proven
without the response body or a reason code.

## Diagnostic contract

The API server now emits one privacy-safe `[BI100 4XX]` record for each Chat
Completions 4xx. It records only:

- fixed reason code;
- status code;
- message, system-message, and tool counts;
- whether an image part is present;
- stream mode and `n`.

It never records message text, image data or URLs, tool definitions or
arguments, exception text, tokens, or generated output.

Known categories are:

- `request_validation_messages`;
- `request_validation_tools`;
- `request_validation_response_format`;
- `request_validation_streaming`;
- `request_validation_generation`;
- `request_validation_sampling`;
- `request_validation_model`;
- `request_validation_other`;
- `request_validation_unknown`;
- `empty_messages`;
- `n_exceeds_max_num_seqs`;
- `unsupported_tool_choice_required`;
- `tool_parser_unavailable`;
- `unclassified_chat_error`.

After service shutdown, `tests/summarize_api_4xx_log.py` reconciles every
Chat Completions 4xx access-log record against exactly one diagnostic record.
It emits `api_4xx_attribution.json` with fixed reason counts and aggregated
request-shape counts. The service gate fails closed when an access-log 4xx is
unattributed, a diagnostic record is orphaned or malformed, or an unknown
reason code appears. The summary contains no raw log lines or request and
response content.

## Qualification rule

- Invalid-request test cases pass only when they return the expected 4xx and
  the service remains healthy.
- `n_exceeds_max_num_seqs` remains a separately reported functional gap. It
  must not be mixed with Agent workload compatibility failures.
- Any 4xx from a valid tool, multimodal, multi-system, reasoning, or 881
  workload request is a real request-success failure.
- Any `unclassified_chat_error` blocks promotion until its fixed request shape
  is reproduced and the response error is diagnosed privately.
- `api_4xx_attribution.rc` must be zero, and its report must have
  `complete=true` with `attribution_delta=0`.
- Aggregate access-log 400 counts cannot be used to claim success or failure
  without phase and reason attribution.

This change is observability-only. It does not alter request validation,
sampling, chat templates, model execution, cache behavior, formal YAML, or
default optimization switches.
