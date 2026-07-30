# M1-165 max_completion_tokens compatibility

## Status

The targeted OpenAI Chat Completions compatibility fix is qualified on the
actual BI100/CoreX runtime overlay. It remains on the private experiment
branch. It does not modify `computility-run.yaml`, runtime defaults, model
weights, tokenizer behavior, chat templates, sampling semantics, or `main`.

This is an API protocol fix. No model or GPU was needed for qualification, and
the result makes no TP4 performance or model-quality claim.

## Root cause and effective path

The Docker build runs `qwen3_6_scripts/patch_ops.sh`, which installs the
repository copies of `protocol.py`, `serving_chat.py`, and `api_server.py`
over the CoreX vLLM package. The effective `ChatCompletionRequest` used
`extra="forbid"` and declared only the legacy `max_tokens` field. Requests
containing `max_completion_tokens` therefore failed Pydantic validation before
tokenization or model execution.

The common serving path computes the context-limited default budget and then
calls `ChatCompletionRequest.to_sampling_params`, or the corresponding beam
conversion. Streaming, non-streaming, tools, reasoning, and multimodal
requests all pass through this conversion before their response-specific
branches.

## Compatibility rule

`max_completion_tokens` is now an optional integer with the same Pydantic type
and downstream `SamplingParams` boundary validation as `max_tokens`.

The effective budget is selected in this order:

1. non-`None` `max_completion_tokens`;
2. non-`None` legacy `max_tokens`;
3. the existing context-derived default.

This makes the conflict rule explicit: when both fields are present,
`max_completion_tokens` wins. An explicit zero or negative value does not fall
back to `max_tokens`; it reaches the existing sampling validation and returns
400. No new request-length cap is introduced.

The rule follows the current vLLM explicit non-`None` selection and the OpenAI
deprecation of `max_tokens` in favor of `max_completion_tokens`. vLLM v0.8.5
also routes both names to the same sampling budget:

- https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/entrypoints/openai/protocol.py
- https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/protocol.py
- https://github.com/openai/openai-python/blob/main/src/openai/types/chat/completion_create_params.py

## Broader field audit

The offline contract compares the local request model against:

- vLLM v0.8.5 commit
  `ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d`;
- openai-python commit
  `3844843c277f42b0b18beaa58152cfda61df524a`.

Every reference-only top-level field must be explicitly classified. The
current result covers five vLLM-only and sixteen OpenAI-only fields.
Unsupported semantic features such as `reasoning_effort`, `service_tier`,
deprecated `functions`, hosted web search, and explicit prompt-cache controls
remain forbidden. The audit does not auto-accept or silently ignore new
fields, and submission preflight now runs it.

## Local verification

The focused protocol, telemetry, audit, probe, evidence, and preflight suite
passed 56 tests. Submission preflight passed all ten checks, including the
new compatibility audit. The complete tracked unit suite passed 1424 tests
with 13 environment-dependent skips.

The commands were:

```bash
python3 -m unittest \
  tests.test_protocol_unit \
  tests.test_api_4xx_telemetry_unit \
  tests.test_chat_request_compat_field_audit_unit \
  tests.test_max_completion_tokens_runtime_probe_unit \
  tests.test_m1_165_max_completion_tokens_evidence_unit \
  tests.test_submission_preflight_unit
python3 tests/submission_preflight.py
python3 -m unittest discover -s tests -p 'test_*_unit.py'
```

The local worktree also contained an unrelated, untracked M1-164 test from
another experiment. It was excluded from the local discovery invocation and
is not part of these commits; a clean checkout contains the 1424 tracked
tests reported above.

## Runtime verification

The immutable overlay was built from clean source
`5863f7ae05657181c36716dd377108d52f4d524f` on `ssh-73ca29ba`.

| Component | Result |
| --- | --- |
| vLLM | `0.6.3+corex.3.2.3` |
| Torch | `2.1.0+corex.3.2.3` |
| Transformers | `4.55.3` |
| Overlay install | qualified, exit 0 |
| Runtime tree SHA-256 | `1c2e10ebef85d21ffb548ed027e26c1aca2d88a3d6e2f68e23445db968a7828f` |
| API overlay identity | all three source and installed SHA-256 values equal |
| Synthetic API cases | 10/10 matched |
| HTTP statuses | seven 200, three expected 400, zero 500 |
| Residual probe processes | zero |
| Fatal scan | zero matches |

The seven accepted cases cover streaming, non-streaming, tools, multimodal,
reasoning switch, legacy `max_tokens`, and both-field precedence. Each entered
the serving conversion and exposed the expected final
`SamplingParams.max_tokens`. Invalid type, zero boundary, and an unrelated
unknown field returned 400 as designed.

The request-validation logs contained only bounded counters plus
`validation_field` and `validation_type`. Scans found no synthetic request
value, message content, tool arguments, or image data.

The first probe harness used `httpx`, but the immutable base runtime lacks its
optional `idna` dependency. It failed before running any case and is excluded
from qualification. The final harness uses a standard-library ASGI client and
adds no package dependency.

Privacy-safe evidence is in
`docs/experiments/evidence/M1_165_MAX_COMPLETION_TOKENS_COMPAT_20260730`.

## Commits

- `5863f7a`: runtime protocol support, overlay identity checks, request tests,
  and the original probe;
- `7fdf205`: pinned vLLM/OpenAI field compatibility audit in submission
  preflight;
- `2862d88`: dependency-free ASGI runtime probe.

No formal YAML or `main` promotion is part of M1-165.
