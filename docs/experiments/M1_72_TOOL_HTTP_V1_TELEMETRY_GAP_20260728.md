# M1-72 tool HTTP v1 telemetry gap

Date: 2026-07-28

## Result

The first real-service M1-72 run is intentionally preserved as negative
evidence. Both service arms passed their complete nine-case HTTP probes, but
the outer comparison failed closed because two expected candidate 400
responses were classified as `request_validation_unknown`.

The run used physical GPU1 on `ssh-73ca29ba`, the four-layer real-weight
Qwen3.6 diagnostic checkpoint, `max_model_len=262144`, and immutable runtime
overlays bound to:

| Arm | Source revision | Port |
| --- | --- | ---: |
| control | `c78d55d0a7637baf4910af68b6d6ba4e286a1254` | 8021 |
| candidate | `cdb1bc41f728a5610a3632ad7923d73a90748919` | 8022 |

The runtime-pair verifier passed with exactly the expected three-file delta:
`api_server`, `chat_utils`, and `protocol`.

## Behavior evidence

The candidate produced HTTP 200 for both concrete compatibility gaps:

- an explicit `strict=false` function definition;
- object-form assistant tool-call arguments in conversation history.

The control reproduced HTTP 400 for both forms. Candidate outputs were exact
against their canonical accepted forms, and canonical tool requests remained
exact across the two overlays.

Malformed JSON tool arguments, unsupported `strict=true`, and unsupported
`tool_choice=required` all remained HTTP 400. The service returned HTTP 200
after those expected failures. This run therefore found no model execution,
request-semantics, or service-health regression in the tested scope.

## Failed gate

The candidate emitted three privacy-safe 4xx markers. They reconciled
one-for-one with the three HTTP 400 responses, but their reason distribution
was:

```text
request_validation_tool_strict = 1
request_validation_unknown = 2
```

The required distribution was:

```text
request_validation_tool_strict = 1
invalid_tool_arguments_json = 1
unsupported_tool_choice_required = 1
```

Both unknown records come from model-level Pydantic validators whose error
location is empty. The current classifier uses only field locations, so it
cannot distinguish them. This is an observability defect, not a reason to
relax the comparison gate.

The bounded fix is to recognize only the exact fixed validator messages
already defined by the request model and map them to fixed enums. Unknown
messages must remain `request_validation_unknown`, and raw validation text
must never enter logs or evidence.

## Resource integrity

Both arms and the outer runner passed process-group cleanup, service
postflight, repeated GPU compute preflight, free-memory comparison, and
fatal/timeout scans. There were no residual API servers, workers, or GPU
holders. GPU1 began and ended with 34,057,748,480 free bytes and the same
1,073,741,824 matmul checksum.

## Decision

The request compatibility behavior is suitable for a controlled rerun after
the specific telemetry fix. This v1 run is not qualified overall and does not
authorize a default switch, `computility-run.yaml` change, `main` merge, or
production promotion.

It does not evaluate full-model semantics, TP4, long-context correctness,
performance, the official 881 requests, or competition score.

Evidence:

```text
docs/experiments/evidence/M1_72_TOOL_HTTP_V1_TELEMETRY_GAP_20260728
```
