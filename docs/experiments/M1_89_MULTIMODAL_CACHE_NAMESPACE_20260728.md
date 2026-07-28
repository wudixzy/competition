# M1-89: multimodal prefix-cache namespace correctness

## Status

`LOCAL CORRECTNESS GATES PASSED; COREX SERVICE EVIDENCE PENDING`.

The initial runtime implementation is commit `0553769`; the installed-overlay
gate was added in commit `8a39916`. Commit `369ff5d` corrects the empty
multimodal-container boundary and upgrades the gate to v2. All are on the
private branch
`fix/M1-89-multimodal-cache-namespace-20260728`. They do not authorize a
`main`, submission YAML, runtime-default, or repository-visibility change.

## Problem

The content-addressed prefix cache already separates text tokens by a runtime,
adapter, and multimodal namespace. The local audit found five correctness
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
5. The real `Sequence.multi_modal_data` property returns `{}` for a text-only
   request. The initial `is not None` correction therefore placed every text
   request in an empty multimodal namespace, while the first runtime gate used
   a synthetic `None` value and asserted the opposite behavior.

These are cache correctness and availability defects. They do not justify
claiming a performance gain.

## Implementation

- The block manager reads multimodal data once and never evaluates its truth
  value. `None` and an empty `Mapping` are treated as no multimodal payload;
  non-empty mappings and unknown objects continue through content
  normalization or request-local isolation.
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
- `tests/qwen36_cache_namespace_runtime_gate.py` v2 loads the actual installed
  block manager, `Sequence` implementation, and real Pillow implementation.
  Its empty-container check obtains `{}` from the installed
  `Sequence.multi_modal_data` property instead of synthesizing it. It repeats
  nine fixed checks and writes only booleans, bounded exception types,
  source/runtime identity, and module SHA-256. Import or initialization
  failure produces a redacted structured failure report instead of relying
  on a traceback.
- Related cache, scheduler, gate, and runtime-identity tests: 50 passed,
  2 dependency skips.
- Full tests-root discovery after the v2 correction: 1028 passed,
  25 dependency skips.
- Submission preflight: all 9 checks passed.
- Quality data and 53-case metric manifests passed.
- Python/shell syntax and `git diff --check` passed.

The local environment has no CoreX GPU and no Pillow installation. These
results prove source-level behavior only; they do not prove service-level
multimodal inference, cold/warm output identity, latency, or throughput.

## Offline evidence boundary

A bounded audit of repository results and structured evidence found no private
cache trace with complete request ordinals `1..881` under the v4 trace
contract. The platform `main` result contains aggregate counters only. It
cannot support per-request residual-prefill projection or an honest
`fine32`/`admission64` comparison, so `admission64` remains unqualified.

The OpenAI usage `cached_tokens` path is the live-KV/exact-GDN-restore
intersection selected by the scheduler. The allocator's prefix-cache hit-rate
counter still measures raw KV allocator hits. These are different metrics;
neither should be relabeled or changed until the evaluator's metric source is
identified from an attested run.

## Required remote gate

The latest lightweight monitor attempted the declared `ssh-73ca29ba` endpoint
three times with a 12-second bound. Every attempt failed in the TLS
ProxyCommand layer with `Connection closed by UNKNOWN port 65535`; no remote
command or GPU operation ran.

On a recovered BI100 host, install an immutable overlay built from the exact
current branch revision. First run
`tests/verify_bare_host_runtime_identity.py`, then run the non-model gate from
outside the repository:

```bash
SOURCE_REVISION=$(git -C /path/to/source rev-parse HEAD)
RUNTIME=/path/to/immutable/site-packages

cd /tmp
PYTHONPATH="$RUNTIME:/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages" \
LD_LIBRARY_PATH="/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib" \
python3 /path/to/source/tests/qwen36_cache_namespace_runtime_gate.py \
  --runtime-site-packages "$RUNTIME" \
  --source-revision "$SOURCE_REVISION" \
  --out /tmp/m1-89-cache-namespace-runtime-gate.json
```

Only after runtime identity and all nine installed-overlay checks pass may the
fixed single-GPU service A/B run. It must check:

- pure text and the runtime's empty multimodal mapping use the same namespace;
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
