# M1-86 multi-image single-GPU A/B

Date: 2026-07-28

## Objective

The historical official aggregate reported 22 image-related 4xx responses
among 27 image requests. That aggregate predates several compatibility fixes,
so it is motivation for a controlled experiment, not proof that the current
runtime still has the same failure.

M1-86 tests whether allowing two images per prompt removes a valid multi-image
rejection without changing one-image behavior, cache identity, request
semantics, or reference compute. It does not modify `computility-run.yaml` or
a default setting. It adds diagnostic-only fields to the optional
`BI100_CACHE_TRACE=1` path; both arms use that exact same instrumentation, and
the production path with tracing disabled is unchanged.

## Fixed experiment

`scripts/run_m1_86_multi_image_ab.sh` runs two fresh services on one healthy
BI100 in fixed order and reuses the same port only after verified cleanup:

1. control: the vLLM default image limit of one;
2. candidate: the sole command suffix
   `--limit-mm-per-prompt image=2`.

Both arms use the same source revision, immutable runtime overlay, four-layer
structural diagnostic checkpoint, model source, environment, port, and
reference kernels. They keep `max_model_len=262144`, block size 16, prefix
caching, `fine32/direct`, and full-attention KV accounting.

The fixed 13-case streaming HTTP matrix checks the model capacity contract,
one-image cold inference, two-image initial and warm inference, reversed image
order with its own warm replay, three indexed-PNG cold/warm pairs, and
post-request health. The indexed variants use identical dimensions and pixel
indices while changing either palette bytes or transparency metadata. Requests
use `temperature=0`, seed `20260728`, streaming usage, and thinking disabled.

## Qualification contract

The control must reject only the second image with one fully attributed
`image_count_limit` 400 response. The candidate must accept both images with
no 4xx response. One-image output summaries must be exact across arms.
Candidate two-image initial and warm summaries must be exact, as must the
reversed-image initial and warm summaries. Both warm requests must report
effective cached tokens. In both arms, every indexed-PNG cold request must
report zero effective cached tokens, its warm replay must report a positive
effective hit, and all deterministic generation fields must be exact within
the pair and across arms.

For the normal/reversed two-image pair, `cached_tokens == 0` is not used as the
sole isolation criterion because blocks before the first multimodal difference
may be reused legitimately. Instead, the runner parses privacy-safe
`BI100_CACHE_TRACE` v4 records. Normal and reversed prompts must have different
SHA-256 block chains; each request's initial raw KV hit, effective GDN hit, and
observed cached-token count must stay within the longest content-identical
block prefix available from earlier requests.

The indexed variants exercise the stronger M1-89 invariant: multimodal content
is part of the first block namespace. Their three cold chains must therefore be
distinct, have zero common leading blocks with all prior requests, and restore
zero KV/GDN blocks. Each warm chain must exactly match its cold chain and
restore at least one GDN-backed block. The actual GDN restore digest must equal
the chain digest at that exact boundary. The initial raw-hit field is frozen
before later chunked prefill steps, so a request's own newly computed chunks
cannot be mistaken for cross-request reuse. HTTP and trace accounting must
agree exactly. The v2 trace contract expects 7 records from control and 11 from
candidate and stores no raw image, token, prompt, or output.

Each arm must pass the exact lifecycle gate set: preflight, port availability,
service contract, startup, capacity, probe, scoped cleanup, v4 cache-trace
qualification, privacy-safe 4xx attribution, fatal scan, service postflight,
post-run preflight, and preflight comparison. Missing or extra lifecycle
fields fail closed. Status SHA-256 values bind the exact probe, trace,
attribution, capacity, service-contract, startup, process identity, service
postflight, and GPU preflight-comparison inputs consumed by the aggregate
decision. The contract's `CUDA_VISIBLE_DEVICES`, each preflight comparison, and
each service postflight must all identify the runner's declared physical GPU.
The candidate may lose no more than two percent of GPU blocks and both arms
must retain at least the 16,384 blocks required by the 262,144-token capacity
contract.

Before the API server is executed, a small launcher creates a new session and
atomically records its PID, PGID, SID, `/proc` starttime, and a private session
token. The process identity must match the background leader and is bound into
the aggregate evidence. Service startup uses one monotonic absolute deadline
and verifies the same starttime. Cleanup sends SIGTERM to only that recorded
process group, waits 60 seconds, escalates only verified survivors, and
waits/reaps. Both TERM and KILL paths revalidate the recorded starttime and
private token before signalling. Repeated TERM/INT is ignored while cleanup is
active. The outer trap repeats process and GPU postflight checks and scans
fatal, Gloo, NCCL, worker-loss, and timeout evidence.

Reports contain response summaries and SHA-256 digests only. They do not store
prompts, image data URLs, generated text, or raw user content.

## Current status

The v2 HTTP/trace/A-B contract is implemented in private commit `d5f9b85`.
Focused CPU-only tests passed 53 of 53. No GPU result has been recorded: the
current SSH proxy closed before authentication during the latest bounded
probe, and the local host has no usable CoreX GPU runtime or Pillow package.
The PNG generator itself is validated for chunk CRCs, identical indexed pixel
bytes, distinct palette metadata, and distinct transparency metadata.

The four-layer checkpoint establishes parser, cache-isolation, capacity, and
lifecycle behavior only. It cannot establish full-model semantic quality,
official 881-request success rate, TP4 performance, or production readiness.

## Invocation after GPU recovery

Install an immutable overlay from the exact committed revision, then run:

```bash
export BI100_RUNTIME_SITE_PACKAGES=/absolute/path/to/immutable/site-packages
GPU_INDEX=<healthy-index> PORT=8030 \
  scripts/run_m1_86_multi_image_ab.sh \
  private-bi100-instance \
  /tmp/m1-86-multi-image-YYYYMMDDTHHMMSSZ
```

Raw logs and responses remain under the private `/tmp` run root and must not
be committed. Only privacy-safe structured evidence may be retained.

## Interpretation

A qualified M1-86 result authorizes only the single-GPU structural diagnostic
phase. It does not authorize changing the default image limit, formal YAML,
`main`, or production behavior. A full-model TP4 quality and performance A/B,
followed by restricted 881-request attribution, remains mandatory.
