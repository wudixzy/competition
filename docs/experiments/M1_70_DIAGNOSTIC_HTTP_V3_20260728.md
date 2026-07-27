# M1-70 diagnostic HTTP v3

Date: 2026-07-28

## Objective

M1-70 tests two request-compatibility changes against a frozen CoreX vLLM
baseline by starting real API services with a four-layer, real-weight
Qwen3.6-35B-A3B diagnostic checkpoint:

- preserve equivalent system text represented as text-content parts or
  multiple system messages;
- keep the native multimodal request path functional and distinguish an
  explicit two-image capacity setting from the default one-image setting.

The experiment also verifies deterministic cold/warm cache behavior and
privacy-safe attribution of expected image-count 4xx responses.

## Identity

Runner source:

```text
dfba141669518f554ed72f9372526b7de6bdb0b2
```

Instance and resource:

```text
instance = ssh-73ca29ba
physical GPU = 1
device = Iluvatar BI-V100
```

Runtime overlays:

| Arm | Revision | Port | Image limit |
| --- | --- | ---: | ---: |
| baseline | `cdb1bc41f728a5610a3632ad7923d73a90748919` | 8018 | 1 |
| candidate default | `37001edff643d98bf41bf4a52e0a145329003315` | 8019 | 1 |
| candidate image2 | `37001edff643d98bf41bf4a52e0a145329003315` | 8020 | 2 |

The runtime verifier passed and isolated the candidate delta to
`api_server.py` and `protocol.py`. The full diagnostic-checkpoint hash and
tensor contract passed: four layers, five shards, 424 model tensors, 333
visual tensors, and 11,345,363,552 weight bytes.

All arms retained the same model, tokenizer, `max_model_len=262144`,
`temperature=0`, seed, request order, cache mode, and disabled rejected
compute candidates.

## HTTP results

All three arms passed all eight cases.

| Case | Baseline | Candidate default | Candidate image2 |
| --- | --- | --- | --- |
| model capacity contract | 200, 262144 | 200, 262144 | 200, 262144 |
| canonical system string | 200 | 200, exact | 200, exact |
| one system with text parts | 200, exact | 200, exact | 200, exact |
| multiple system text parts | 400 | 200, exact | 200, exact |
| one image | 200 | 200, exact | 200, exact |
| at-limit replay | one image, exact | one image, exact | two images, exact |
| over-limit image | two images, 400 | two images, 400 | three images, 400 |
| post-4xx health | 200 | 200 | 200 |

The canonical system output SHA-256 is identical across all arms. The
candidate's equivalent system representations also preserve the complete
deterministic generation contract.

The one-image output is identical across all arms. The two-image candidate's
first request records zero cached tokens; its replay records 448 cached tokens
and an identical generated-message SHA-256. No mismatched multimodal cache
state was accepted.

## 4xx attribution

Each candidate arm produced exactly one expected 400 response:

- default limit: two data images;
- explicit image2 limit: three data images.

Both records were reconciled one-for-one as `image_count_limit`, with
`classified=true`, `complete=true`, no malformed markers, and no raw request,
response, tool schema, URL, or image bytes in the saved report.

The old baseline attribution format is retained only as a control placeholder
and is not compared as v3 evidence.

## Cleanup and GPU state

Every arm passed startup, probe, process-group cleanup, fatal scan, service
postflight, repeated GPU preflight, and preflight comparison.

The outer runner also passed all gates. SIGTERM was scoped to the service
process group with a 60-second graceful window; no broad process kill was
used. Each arm and the final postflight observed three consecutive clean
samples with no API server, worker, or GPU holder.

GPU1 began and ended with:

```text
free bytes = 34,057,748,480
total bytes = 34,057,748,480
matmul checksum = 1,073,741,824
free-memory drop = 0
```

Fatal, Gloo, NCCL, worker-loss, timeout, and retained-process scans are empty.

## Decision

The M1-69 system-text and native one-image compatibility changes are qualified
for the next full-model and TP4 quality gates. The explicit two-image setting
is structurally qualified on the diagnostic model, but it is not authorized
as a default change.

This run does not evaluate full-model semantic quality, long-context
correctness, TP4 behavior, throughput, the official 881 requests, or
competition score. It does not authorize a `computility-run.yaml`, default
switch, `main`, or repository-visibility change.

Evidence:

```text
docs/experiments/evidence/M1_70_DIAGNOSTIC_HTTP_V3_20260728
```
