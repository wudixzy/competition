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
- message, system-message, tool-result, assistant-tool-call, and tool counts;
- `strict=false` and `strict=true` tool counts;
- bounded tool-choice kind (`unset`, `none`, `auto`, `required`, `named`, or
  `other`);
- whether an image part is present;
- stream mode and `n`.

It never records message text, image data or URLs, tool definitions or
arguments, exception text, tokens, or generated output.

Validation categories distinguish:

- message role, content, tool calls, and tool-call ID;
- tool definition, parameters, `strict`, and tool choice;
- response format, streaming, generation, sampling, model, and other fields.

Runtime categories include:

- `empty_messages`;
- `invalid_tool_arguments_json`;
- `invalid_tool_arguments_type`;
- `n_exceeds_max_num_seqs`;
- `unsupported_tool_choice_required`;
- `tool_parser_unavailable`;
- `unclassified_chat_error`.

`tests/summarize_api_4xx_log.py` reconciles every Chat Completions access-log
4xx with exactly one marker after the service has exited. Missing, orphaned,
malformed, unknown, or unclassified records fail the quality runner. Its v2
report contains only aggregate reason and request-shape counts.

## Compatibility fixes

M1-68 closes two request-shape failures without weakening request semantics:

- explicit OpenAI function-tool `strict=false` is accepted as a no-op and
  excluded from the tool dictionary passed to the chat template;
- assistant tool-call history may provide `function.arguments` as either a
  JSON object or a JSON-encoded object string; both reach the template as the
  same object.

`strict=true` and `tool_choice=required` remain fail-closed because the CoreX
vLLM 0.6.3 path does not implement their full semantics. Invalid JSON, arrays,
and scalar tool arguments also remain rejected.

## Qualification rule

- Invalid-request test cases pass only when they return the expected 4xx and
  the service remains healthy.
- `n_exceeds_max_num_seqs` remains a separately reported functional gap. It
  must not be mixed with Agent workload compatibility failures.
- Any 4xx from a valid tool, multimodal, multi-system, reasoning, or 881
  workload request is a real request-success failure.
- Any `unclassified_chat_error` blocks promotion until its fixed request shape
  is reproduced and the response error is diagnosed privately.
- `api_4xx_attribution.rc` must be zero, with `complete=true`,
  `classified=true`, and `attribution_delta=0`.
- Aggregate access-log 400 counts cannot be used to claim success or failure
  without phase and reason attribution.

These compatibility fixes do not alter sampling, model execution, cache
behavior, formal YAML, or default optimization switches.
