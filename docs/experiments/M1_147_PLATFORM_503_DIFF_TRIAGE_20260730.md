# M1-147 platform 503fa7c result and current-code triage

Date: 2026-07-30

## Inputs

- Platform result source revision:
  `503fa7c670b6172d9a3e2912166e78317f5e289f`.
- Result file:
  `result/result0730/503fa7c670...`, SHA-256
  `28deb18cca59a2bbec76388871572d8c933bffe498f1f0a5284ba294b363ee06`.
- Current stable main:
  `fb0084fc778e62c26d6a6e108b87dc027ae2ed79`.
- Current diagnostic run:
  `/tmp/m1-147-fb0084f-diagnostic-gpu1` on `ssh-73ca29ba`, GPU 1,
  TP1, using the verified four-layer real-weight diagnostic checkpoint.

The platform result is evidence for the exact `503fa7c` submission. It is not
silently relabeled as a result for `fb0084f`.

## Platform observation

The `503fa7c` result reports:

- functional pass rate 50/52, with `n=2` and base64 multimodal failed;
- 631 successful requests out of 881 and 250 request errors;
- 226 tool-request 4xx, 22 image-request 4xx, and 7 multi-system 4xx;
- cache hit rate 0.54;
- Output TPS P10 4.42 and mean 12.90;
- TTFT P90 27.488 seconds;
- the run timed out.

The earlier `result/20260728` artifact has the same 631/881 success result and
the same functional failures, but reports cache hit rate 0.59, Output TPS P10
5.61, and TTFT P90 14.529 seconds. Mean Output TPS is effectively unchanged at
12.91. This repeat variance prevents a single platform run from adjudicating
small performance changes.

## Code-difference attribution

`computility-run.yaml` is byte-identical between `503fa7c` and `fb0084f`
(SHA-256
`8d67b1c4cce264429e95a1cfeeb5342a01a6a97f843094fcef552b2c79690fc4`).
The production TP4, 262144 context, 8192-token chunk, and request-concurrency
contract therefore did not change.

The relevant runtime changes are:

1. `2014b7e` is a real default-path correctness fix. It caps a physical
   admission64 prefill step at the pending GDN capture boundary, so cold
   capture and warm replay use the same recurrent-state partition. It closes
   the observed 235K `5616 + 8` boundary mismatch.
2. `a534a05` adds privacy-safe request-validation field/type attribution. It
   does not relax request schemas and cannot itself reduce 4xx counts.
3. `f39fd69` fixes duplicate named-tool identity in an accepted streaming
   response after a zero-token delta. It does not change request validation.
4. M1-109 fused-prefill work and its later numeric/quality harnesses remain
   behind `BI100_ATTN_COREX_FUSED_PREFILL=0` by default. Their 7%-9% 235K
   improvement must not be credited to the current formal main.
5. The deterministic sequential `n=2` path (`383381c`) was already an ancestor
   of `503fa7c`. The platform `n=2` failure therefore is not proven to have
   been fixed by a post-503 runtime change.
6. The Qwen chat and multimodal parsing files used by the failing base64 case
   have no post-503 behavioral fix. The platform base64 failure is likewise
   not proven to have been fixed by the commit difference alone.
7. The M1-68 `strict=false` and tool-history normalization fixes were already
   in `503fa7c`. They cannot explain away the remaining 226 tool 4xx.

The Dockerfile changed to the required current Harbor base and gained runtime
overlay verification. That affects build/runtime identity, but does not by
itself establish a model-serving behavior change.

## Current-shape diagnostic

M1-147 used the exact `fb0084f` runtime overlay. It is a protocol and model
plumbing diagnostic, not a full-model capability or TP4 performance result.

Passed:

- the `n=2` request returned HTTP 200 with choice indices 0 and 1,
  deterministic equal choices, prompt usage counted once, and completion
  usage summed across both choices;
- one-image base64 input returned HTTP 200, deterministic replay matched, a
  two-image over-limit request returned the expected classified HTTP 400, and
  service health remained HTTP 200;
- the 10-case request contract, 8-case compatibility gate, and 13-case tool
  schema/history/streaming gate all passed;
- partial and warm prefix accounting reported 8176 and 11600 cached tokens;
- runtime identity, cleanup, postflight, repeated GPU preflight, fatal scan,
  and timeout scan passed.

The overall diagnostic did not qualify because the four-layer checkpoint
failed the tool-choice semantic gate:

- omitted and `auto` modes returned HTTP 200 but did not finish as tool calls;
- named non-stream returned HTTP 200 but generated invalid JSON arguments;
- named stream returned HTTP 200 but failed the stream semantic contract.

All six valid tool-choice requests reached the server and returned HTTP 200.
These failures therefore do not reproduce the platform request-validation
4xx. They also cannot establish full-model tool-call capability because the
diagnostic checkpoint has only four model layers.

## Revised interpretation

- The current canonical `n=2` and base64 request shapes work on `fb0084f`.
- The platform failures are not reproducible in this scoped diagnostic, but
  they are not attributable to a specific post-503 fix. Runtime-overlay
  identity, platform request shape, or another workload variant remains a
  plausible cause.
- The GDN capture-boundary defect is fixed and separately covered by full TP4
  long-context evidence from M1-125.
- The 226 tool 4xx remain unresolved as a workload-level issue. The next
  platform run must use the new aggregate reason, `validation_field`, and
  `validation_type` diagnostics before any compatibility change is proposed.
- Current main should not be assigned the fused-prefill performance gain.
  M1-109 must pass the hash-bound L2/L3 funnel and a fresh full-model TP4 A/B
  before any production default or YAML change.

No result in this triage authorizes a formal YAML change or production
promotion.
