# M1-89: multimodal prefix-cache namespace correctness

## Status

`LOCAL CORRECTNESS GATES PASSED; COREX SERVICE EVIDENCE PENDING`.

The implementation is commit `0553769` on the private branch
`fix/M1-89-multimodal-cache-namespace-20260728`. It does not authorize a
`main`, submission YAML, runtime-default, or repository-visibility change.

## Problem

The content-addressed prefix cache already separates text tokens by a runtime,
adapter, and multimodal namespace. The local audit found four correctness
gaps in the multimodal namespace:

1. `if seq.multi_modal_data` could invoke unsafe or ambiguous truth-value
   conversion before normalization.
2. Only `TypeError` triggered request-local isolation. A supported image or
   tensor whose normalization raised a bounded runtime error could fail the
   request instead.
3. A PIL palette image hashed its mode, dimensions, and pixel indices, but not
   its palette or transparency. Equal indices with different rendered content
   could therefore share a cache namespace.
4. Request-local fallback salts and warning state were retained indefinitely.
   Reusing a request ID could reuse an old isolation namespace.

These are cache correctness and availability defects. They do not justify
claiming a performance gain.

## Implementation

- Multimodal presence is checked with `is not None`; object truthiness is
  never evaluated.
- Normalization failures in the bounded expected exception set fall back to a
  stable per-request random namespace. The request continues without
  cross-request prefix reuse.
- PIL image namespaces now include palette mode, palette values, and
  transparency metadata in addition to mode, dimensions, and raw image bytes.
- The block manager exposes `release_request_cache_namespace(request_id)`.
- The production scheduler calls this release hook for decoder-only and
  encoder-decoder groups on the shared finished, aborted, and async-stopped
  cleanup path. The hook remains optional for older block managers.
- `admission64.repeated_branch_candidate` now returns no candidate for an
  empty live-prefix sequence instead of indexing an empty list.
- The install comment now states that reported `cached_tokens` is the
  intersection of live KV and an exact GDN restore state.

No model weights, dtype, tokenizer, chat template, request sampling semantics,
tool/reasoning/multimodal capability, context limit, or formal command changed.

## Local evidence

- Focused prefix, scheduler, and GDN tests: 38 passed, 1 pre-existing optional
  Pillow integration test skipped.
- New mandatory fake-image tests execute without Pillow and cover palette
  isolation, transparency isolation, image-read failure fallback, ambiguous
  truthiness, request-ID reuse after release, and physical block reuse.
- A fixed-seed 1000-step state machine compares `fine32`, `admission64`, and
  `off` policy state against an independent `OrderedDict` LRU oracle.
- Related cache/static/runtime-identity tests: 89 passed, 1 dependency skip.
- Full tests-root discovery: 1018 passed, 25 dependency skips.
- Submission preflight: all 9 checks passed.
- Quality data and 53-case metric manifests passed.
- Python/shell syntax and `git diff --check` passed.

The local environment has no CoreX GPU and no Pillow installation. These
results prove source-level behavior only; they do not prove service-level
multimodal inference, cold/warm output identity, latency, or throughput.

## Required remote gate

On a healthy BI100 host, install an immutable overlay built from exactly
`0553769`, then run a fixed service A/B that checks:

- the same image and prompt produce identical cold/warm token output;
- equal palette images reuse the same namespace;
- equal pixel indices with different palettes or transparency do not reuse;
- two different images never restore the same KV/GDN prefix;
- an injected normalization failure uses request-local isolation while the
  multimodal request still completes correctly;
- namespace state is absent after normal completion, abort, and async stop;
- usage `cached_tokens` never exceeds the exact KV/GDN-restorable prefix;
- no 5xx, fatal, OOM, Gloo/NCCL reset, worker loss, timeout, or process leak.

The runner must clean only its attested process group, send `SIGTERM`, wait
60 seconds, use `SIGKILL` only for survivors, wait/reap, and require clean GPU
preflight and postflight. Passing this gate is still only a correctness
qualification; TP4 quality and performance A/B remain mandatory before any
promotion.
