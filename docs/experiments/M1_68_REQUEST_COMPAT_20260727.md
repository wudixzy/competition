# M1-68 request compatibility

Date: 2026-07-27

## Motivation

The latest reported platform run completed 631 of 881 requests and returned
250 errors. Its aggregate counters included 226 tool-request 4xx responses,
22 image-request 4xx responses, and seven multi-system 4xx responses. Those
sets may overlap, and the platform report did not include privacy-safe reason
codes. This experiment therefore does not claim that all tool errors share one
cause.

The saved 2026-07-13 Docker log has 29 Chat Completions 400 responses:

- 23 occurred before the engine fatal;
- six occurred after the prefix block-table fatal and are secondary failures;
- one pre-fatal response has the known Qwen image model-type exception;
- one has the known empty-message template `IndexError`;
- the remaining old access records have no request-safe reason marker.

Both known exceptions and the block-table fatal are already fixed in the
current lineage. The unattributed records cannot be used as evidence for a new
compatibility change.

## Reproduced gaps

The target runtime uses OpenAI SDK 1.108.2, Pydantic 2.11.9, and CoreX vLLM
0.6.3. Local protocol tests previously replaced the SDK message types with
plain dictionaries and therefore missed lazy Pydantic validation.

Two synthetic, content-free request shapes reproduced failures:

1. OpenAI function definitions with explicit `strict=false` were rejected by
   the local protocol's `extra="forbid"` model.
2. Assistant tool-call history with `function.arguments` already represented
   as a JSON object passed the outer request model but failed when Pydantic's
   lazy `ValidatorIterator` was consumed.

The second issue is relevant to Agent request converters that retain parsed
tool arguments between turns. It is an equivalent representation of the JSON
object, not permission to ignore or rewrite tool history.

## Implementation

The branch is `diag/M1-68-request-compat-20260727`.

- `ffb28e4` accepts `strict=false`, excludes it from the tool dictionary sent
  to the tokenizer, and rejects `strict=true`.
- `6c7c9db` added object-aware tool history processing but failed the real
  lazy-validator gate. It is retained as negative evidence, not a candidate.
- `f5bdcb0` adds bounded request-validation reason codes and v2 4xx
  reconciliation to the quality runner.
- `cdb1bc4` normalizes object arguments before SDK union validation,
  materializes lazy tool-call iterables, validates that strings decode to JSON
  objects, and rejects malformed JSON, arrays, and scalar values.

`tool_choice=required` remains rejected because the runtime does not implement
its full semantics. `strict=true` remains rejected because accepting it without
constrained tool decoding would silently weaken the request.

No sampling parameter, tokenizer, chat template, model weight, precision,
context limit, formal YAML, or default performance switch changed.

## Exact runtime

Candidate source:

```text
cdb1bc41f728a5610a3632ad7923d73a90748919
```

Immutable overlay:

```text
/root/m1-68-runtime-cdb1bc4
runtime_tree_sha256 =
5196f4030ddc23a716f1f8d89f2c9967aabf3d2c90f361244b8062e37d563d8c
```

The installer reports `qualified=true`, exact source identity, and byte-equal
installed/source `protocol.py`, `chat_utils.py`, and `api_server.py`.

## Runtime gates

The real protocol and preprocessing matrix passed 11/11:

- ordinary text and function tools;
- explicit `strict=false`;
- fail-closed `strict=true` and `tool_choice=required`;
- null-content assistant tool history;
- tool-result text parts;
- image URL request shape;
- multiple string system messages;
- object tool arguments;
- malformed JSON rejection.

The Qwen3.6 tokenizer boundary gate passed all five checks:

| Pair or guard | Result |
| --- | --- |
| JSON string vs object tool history | 316 vs 316 tokens, identical SHA-256 |
| omitted `strict` vs `strict=false` | 268 vs 268 tokens, identical SHA-256 |
| `strict` forwarded to template | no |
| `strict=true` | rejected |
| `tool_choice=required` | rejected |

The old `c78d55d` overlay fails the same probes while consuming the SDK's lazy
tool iterator and rendering tool history. The intermediate `6c7c9db` report
also fails two of 11 cases, proving that the first object-aware patch was not
sufficient.

These gates load the real tokenizer but not model weights. They do not use a
GPU and do not establish end-to-end request success or performance.

## Four-card status

The 2026-07-27 preflight still fails before TP4 service startup:

| GPU | Result |
| --- | --- |
| 0 | timeout at `mem_get_info`; SIGTERM; reaped |
| 1 | pass; 34,057,748,480 bytes free; checksum 1,073,741,824 |
| 2 | timeout at `mem_get_info`; SIGTERM; reaped |
| 3 | timeout at `mem_get_info`; SIGTERM; reaped |

No SIGKILL or broad `pkill` was used. Postflight observed three consecutive
clean samples with no API server, worker, or GPU process.

## 4xx evidence contract

The server now logs only bounded counts and enums for every chat 4xx:

- validation category;
- message, system, tool-result, assistant-tool-call, and tool counts;
- strict true/false counts and tool-choice kind;
- image presence, stream mode, and `n`.

`tests/summarize_api_4xx_log.py` reconciles those markers against access-log
4xx responses after graceful service shutdown. Missing, malformed, unknown, or
unclassified markers make the experiment invalid. No prompt, response, tool
schema, arguments, image URL, image bytes, or raw error text enters the report.

## Decision

Keep `cdb1bc4` as the M1-68 protocol/tokenizer-qualified candidate. Do not
merge it to `main` and do not change `computility-run.yaml` yet.

Required next evidence:

1. healthy TP4 preflight;
2. exact-overlay API service startup;
3. valid `strict=false` and object-history HTTP 200 tests plus invalid JSON
   4xx and post-error health;
4. full functional and Agent workload gates;
5. no fatal, timeout, worker loss, or postflight residue;
6. same-source performance comparison proving no material regression;
7. a platform or restricted-workload run with v2 reason attribution before
   estimating how many of the 250 historical errors this change removes.
