# M1-86 control ErrorResponse shape fix

Date: 2026-07-28

## Observed run

- source revision: `6dbb8e1ba3e5f2e003f4d997b33174e88afa20f5`;
- instance: `ssh-73ca29ba`, physical GPU 1;
- run root:
  `/tmp/m1-86-multi-image-6dbb8e1-gpu1-20260728T065819Z`;
- diagnostic checkpoint:
  `/root/shared-storage/models/Qwen/Qwen3.6-35B-A3B-diagnostic-4L-real`;
- runtime overlay: `/root/m1-94-runtime-6dbb8e1/site-packages`.

The service behaved as the control arm required:

- one-image streaming returned HTTP 200;
- two-image streaming returned the expected HTTP 400 with
  `reason=image_count_limit`;
- the post-request health check returned HTTP 200;
- identical synthetic indexed images had 2,976 effective cached tokens and
  exact cold/warm generation;
- palette and transparency variants had zero cross-variant cached tokens;
- no fatal, timeout, Gloo/NCCL, worker-loss, or residual-process failure was
  found.

The run nevertheless failed `stream_two_images_cold`. The gate accepted only
the nested error shape:

```json
{"error": {"message": "...", "type": "...", "code": 400}}
```

CoreX vLLM 0.6.3 serializes its `ErrorResponse` fields at the top level:

```json
{"object": "error", "message": "...", "type": "...", "code": 400}
```

The HTTP status and service behavior were correct; the harness rejected a
valid OpenAI-compatible error envelope.

## Fix

`qwen36_multi_image_http_gate.py` now accepts either a nested error object or
the top-level `ErrorResponse`. It still requires:

- the exact expected status;
- a non-empty structured message;
- JSON object fields;
- privacy-safe hashes rather than the raw message.

The report records `error_shape=top_level|nested`, so the two paths remain
auditable. The candidate arm's required HTTP 200 behavior is unchanged.

## Validation

- 41 focused M1-86 HTTP, trace, and comparison tests passed;
- Python syntax and `git diff --check` passed;
- a new unit case covers the top-level CoreX/vLLM error shape.

This source fix does not itself qualify M1-86. The control and candidate arms
must be rerun from an exact clean revision before the multi-image cache
candidate can proceed.
