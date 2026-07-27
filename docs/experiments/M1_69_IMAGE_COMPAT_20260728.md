# M1-69 image and system-message compatibility

Date: 2026-07-28

## Motivation

The latest reported platform run on the then-current `main` completed 631 of
881 requests. Its aggregate counters included 226 tool-request 4xx responses,
22 image-request 4xx responses, and seven multi-system 4xx responses. These
sets may overlap, and the report did not attribute individual 4xx responses to
bounded reason codes. This experiment therefore does not claim that the
changes below remove all 250 reported errors.

M1-68 covered tool-schema and tool-history compatibility. M1-69 investigates
two remaining request shapes without changing model behavior:

1. multiple system messages where a system message uses OpenAI text content
   parts rather than a plain string;
2. multi-image parsing and preprocessing boundaries for the Qwen3.6 model.

## Implementation

The private experiment branch is
`diag/M1-69-image-compat-20260728`.

- `ed15ee9` preserves all-text system content parts by joining their text with
  the newline semantics already used by CoreX vLLM `chat_utils`, before the
  existing Qwen multiple-system merge runs. Non-text system parts remain
  untouched and fail through the existing validation/template path.
- `ed15ee9` also upgrades privacy-safe chat 4xx telemetry to v3. It adds
  bounded system-part counts, image counts, image source categories
  (`data`, `remote`, or `other`), and reason codes for known image-count and
  model-type failures. It reads only the first eight URL characters to classify
  a source and records no URL, image bytes, prompt, response, tool schema, or
  tool arguments.
- `37001ed` adds the baseline/candidate tokenizer qualifier that proves the
  system content-part failure and its exact normalization.
- `dd3f1cc` adds a no-weight multi-image preprocessing probe using generated
  128 by 128 red and blue PNG inputs.

No sampling parameter, tokenizer file, chat template, model weight, dtype,
precision, context limit, formal YAML, default performance switch, or default
image limit changed.

## Exact runtime

Runtime-changing source revision:

```text
37001edff643d98bf41bf4a52e0a145329003315
```

Probe-only revision:

```text
dd3f1cc413f32848b4b2996bd2e006823465c700
```

Immutable overlay:

```text
/root/m1-69-runtime-37001ed
runtime_tree_sha256 =
9cb9bf5b21260826372d8f9496a23bc501cb4052ce7c701dbc401be0952f6549
```

`git diff 37001ed..dd3f1cc -- qwen3_6_scripts vllm Dockerfile
computility-run.yaml` was empty. The probe revision therefore did not change
runtime inputs. The installed packages were CoreX vLLM 0.6.3, Transformers
4.55.3, and Torch 2.1.0+corex.3.2.3.

## System-message gate

Both baseline and candidate pass the 12-case request-model matrix. That matrix
alone cannot reveal the system content-part defect because the failure occurs
later at the tokenizer template boundary.

The real Qwen3.6 tokenizer qualifier produces:

| Check | Baseline | Candidate |
| --- | --- | --- |
| System text-parts render | `TemplateError` | pass |
| Canonical merged token count | 32 | 32 |
| Text-parts token count | unavailable | 32 |
| Canonical/text-parts token SHA-256 | unavailable | identical |
| Other template checks | 5/5 | 5/5 |

The candidate digest for both representations is
`7c227b48b70d5d76b494b8bc70af2e2dfcad22c9de6a73303e3c7fab1a505b4d`.
This establishes exact tokenizer input equivalence for the synthetic system
case. It does not establish full HTTP service success.

## Multi-image component gate

The real model config reports `max_model_len=262144` and model type
`qwen3_5_moe`. The default runtime allows one image per prompt and budgets
1,280 multimodal tokens. An explicit diagnostic limit of two accepts two
images and budgets 2,560 tokens.

The no-weight preprocessing probe exercises the real OpenAI request model,
chat parser, Qwen tokenizer, placeholder expansion, input registry, and
multimodal mapper:

| Input | Visual tokens | `pixel_values` shape |
| --- | ---: | --- |
| One synthetic image | 64 | `[256, 1536]` |
| Two synthetic images | 128 | `[512, 1536]` |

All mapped tensors were finite. Repeating the same two images produced
identical token and tensor digests; reversing image order changed both the
processed-token digest and pixel-value digest. The default cap rejected the
second image, while an explicit cap of two accepted it.

This is preprocessing qualification only. Model weights were not loaded, no
API service was started, and no generated multimodal answer was evaluated.
The default remains one image until a healthy TP4 service proves output
correctness, memory headroom, and performance.

## GPU and cleanup status

The four-card preflight could not qualify TP4:

| GPU | Result |
| --- | --- |
| 0 | timeout at `mem_get_info`; SIGTERM; reaped |
| 1 | pass; 34,057,748,480 bytes free; checksum 1,073,741,824 |
| 2 | timeout at `mem_get_info`; SIGTERM; reaped |
| 3 | timeout at `mem_get_info`; SIGTERM; reaped |

GPU 1 passed both the initial and final single-card checks with identical
reports. All scoped children were reaped. Four-card and single-card
postflights each observed three consecutive clean samples with no API server,
worker, or GPU process. The fatal scan found no fatal, Gloo/NCCL reset, worker
loss, or timeout from the qualified single-card run.

Two setup retries corrected the physical GPU ordinal used with
`CUDA_VISIBLE_DEVICES` and completed the CoreX OpenMPI library path. They are
recorded in `run-status-v1.json` and are not counted as candidate results.

## Evidence

The evidence bundle is
`docs/experiments/evidence/M1_69_IMAGE_COMPAT`. `SHA256SUMS` and
`tests/test_m1_69_image_compat_evidence_unit.py` bind:

- exact source and immutable runtime identity;
- baseline and candidate request/tokenizer reports;
- model limits and multi-image preprocessing results;
- failed four-card health plus clean postflight;
- passing GPU 1 preflight/postflight and clean error scan.

The saved reports contain hashes, counts, shapes, package versions, and
synthetic-input metadata. They contain no prompt or response text, image URL
or bytes, tool schema, model weights, credentials, or environment dump.

## Decision

Keep `37001ed` as the runtime-changing M1-69 candidate and `dd3f1cc` as its
probe-only extension. The system text-part normalization and v3 diagnostic
telemetry are component-qualified. Explicit two-image preprocessing is
component-qualified but remains diagnostic.

Do not merge to `main`, change `computility-run.yaml`, or raise the default
image limit yet. Required promotion evidence is:

1. healthy four-card preflight and NCCL/CoreX collective probe;
2. exact-overlay TP4 service startup at `max_model_len=262144`;
3. HTTP system-part, tool, one-image, and multi-image correctness cases;
4. generated multimodal output validation and cold/warm equality where cache
   applies;
5. the complete functional matrix, long-context gates, and capability suite;
6. startup-memory and performance A/B for any image-limit change;
7. restricted-workload v3 4xx attribution before estimating platform impact;
8. clean graceful shutdown, per-card postflight, and fatal scan.
