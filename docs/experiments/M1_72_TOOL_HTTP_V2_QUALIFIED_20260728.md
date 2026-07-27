# M1-72 tool HTTP v2

Date: 2026-07-28

## Objective

M1-72 v2 reruns the frozen real-service tool compatibility A/B after one
bounded observability fix. The fix recognizes exact model-level Pydantic
validator messages whose field location is empty and maps them to fixed
privacy-safe reason enums. It does not change validation rules, HTTP status,
request normalization, chat templates, sampling, model execution, or cache
behavior.

The v1 run proved both request compatibility changes but failed closed because
two expected 400 responses were labeled `request_validation_unknown`.

## Identity

The run used physical GPU1 on `ssh-73ca29ba` and the four-layer real-weight
Qwen3.6 diagnostic checkpoint.

| Arm | Source revision | Runtime tree SHA-256 | Port |
| --- | --- | --- | ---: |
| control | `c78d55d0a7637baf4910af68b6d6ba4e286a1254` | `05d720fa...273b7b` | 8023 |
| candidate | `d2eed78371ef78aee36682c2322fb9ea44ebb5f2` | `6c69c7a...9f4511` | 8024 |

The candidate overlay was atomically installed at
`/root/m1-73-runtime-d2eed78`. Its report is qualified, binds the exact source
revision, records CoreX vLLM 0.6.3, Transformers 4.55.3, and Torch
2.1.0+corex.3.2.3, and confirms that system site-packages were not modified.

The runtime-pair verifier found exactly the authorized three-file delta:
`api_server`, `chat_utils`, and `protocol`. Both arms retained
`max_model_len=262144`, the same model and tokenizer, `temperature=0`, request
order, cache policy, and disabled rejected compute candidates.

## HTTP results

Both arms passed all nine cases.

| Case | Control | Candidate |
| --- | --- | --- |
| model capacity contract | 200, 262144 | 200, 262144 |
| default function tool | 200 | 200, exact across arms |
| explicit `strict=false` | 400 | 200, exact vs omitted |
| JSON-string tool history | 200 | 200, exact across arms |
| object-form tool history | 400 | 200, exact vs string |
| malformed JSON arguments | 400 | 400 |
| unsupported `strict=true` | 400 | 400 |
| unsupported `tool_choice=required` | 400 | 400 |
| health after expected 4xx | 200 | 200 |

The two accepted compatibility forms preserve complete deterministic
generation contracts: generated-message SHA-256, finish reason, prompt and
completion token counts, content/reasoning presence, and tool-call count.

The same four successful tool-generation cases are also exact against the
saved v1 candidate evidence. This is a targeted non-regression result for
those synthetic requests, not a full-model semantic-quality score.

## 4xx attribution

The candidate produced exactly three expected HTTP 400 responses and
reconciled each one to a fixed enum:

```text
invalid_tool_arguments_json = 1
request_validation_tool_strict = 1
unsupported_tool_choice_required = 1
```

There are no unknown or malformed markers. The saved attribution contains no
raw request, response, tool schema, URL, image bytes, log lines, credentials,
or environment values.

Unknown validator messages remain fail closed as
`request_validation_unknown`; the comparison gate was not relaxed.

## Resource integrity

Both arms and the outer runner passed startup, probe, process-group cleanup,
service postflight, repeated GPU compute preflight, free-memory comparison,
and fatal/timeout scans.

Cleanup targeted only each service process group, sent SIGTERM with a
60-second graceful window, used the existing survivor path only if needed,
and waited/reaped the leader. Final postflight found no API server, worker, or
GPU holder.

GPU1 began and ended with:

```text
free bytes = 34,057,748,480
total bytes = 34,057,748,480
matmul checksum = 1,073,741,824
free-memory drop = 0
```

Fatal, Gloo, NCCL, worker-loss, timeout, and retained-process scans are empty.

## Decision

The tested tool request compatibility and privacy-safe attribution changes are
qualified for the next full-model TP4 quality gate. This closes the specific
M1-72 single-GPU diagnostic evidence gap.

The run does not evaluate the full Qwen3.6-35B-A3B model, TP4 correctness,
semantic capability, long context, throughput, the official 881 requests, or
competition score. It does not authorize a default switch,
`computility-run.yaml` change, `main` merge, or repository-visibility change.

Evidence:

```text
docs/experiments/evidence/M1_72_TOOL_HTTP_V2_QUALIFIED_20260728
```
