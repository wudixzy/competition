# BI100 paged-attention decode segmentation fault

Date: 2026-07-14

## Summary

An evaluation process terminated with `Fatal Python error: Segmentation fault`
while executing model decode on BI100/CoreX. This is a native-process memory
fault, not a recoverable Python exception. In TP=4 mode, one worker fault is
enough to terminate the complete model service.

The strongest stack evidence points to the full-attention decode path:

```text
vllm/attention/ops/paged_attn.py:277        forward_decode
vllm/attention/backends/xformers.py:650    forward
vllm/attention/layer.py:101                forward
vllm/model_executor/models/qwen3_5.py      model forward
vllm/worker/model_runner.py:1670           execute_model
```

A second interleaved fault dump shows a thread in:

```text
ixformer/functions/linear.py:67
vllm/model_executor/layers/linear.py
vllm/model_executor/models/qwen3_5.py
```

The nearly identical timestamps and interleaved text strongly suggest that
multiple TP processes or threads failed as part of the same incident.

## What the log proves

- The process received a native segmentation fault during model execution.
- At least one active execution stack was in `paged_attn.forward_decode`.
- Another execution stack was in an ixformer linear operation.
- The failure happened during decode, after service startup and model loading.
- Python exception handling cannot safely recover the affected worker.

## What the log does not prove

- It does not prove that `paged_attn.py:277` first corrupted memory.
- It does not prove that tqdm, uvloop, TensorFlow, Pillow, or the listed Python
  extension modules caused the crash. Faulthandler prints all active threads
  and loaded modules after a fatal signal.
- It does not identify the failing TP rank or the exact request shape.
- It does not show an ordinary out-of-memory failure. A normal OOM generally
  produces a CUDA/CoreX or Torch `RuntimeError` instead of SIGSEGV.

GPU work is asynchronous. A kernel can write out of bounds and the process may
not fail until a later kernel launch or synchronization point. Therefore the
top Python frame is a localization clue, not definitive root-cause proof.

## Ranked root-cause hypotheses

### H1: Invalid paged-attention metadata

The decode kernel may have received an inconsistent block table, context
length, slot mapping, or prefix-cache length. A stale or undersized block table
can cause a native kernel to read an invalid address.

Evidence that would strengthen H1:

- failure occurs only after a prefix-cache hit;
- failure is reproducible at block-size boundaries;
- disabling prefix caching for diagnosis removes the failure;
- cached and uncached requests diverge at the first decode token.

### H2: Unsupported CoreX/ixformer shape or layout

The vendor attention or linear path may not support one particular tensor
shape, stride, alignment, or head dimension even though neighboring shapes
work. This can present as SIGSEGV instead of a checked error in native code.

Evidence that would strengthen H2:

- one fixed context or batch shape always fails;
- making inputs contiguous or changing only the decode length removes it;
- the same failure reproduces without prefix caching.

### H3: Earlier asynchronous memory corruption

An earlier custom or vendor kernel may have written outside its output buffer.
The fault could then surface in `forward_decode` or the next ixformer linear
operation.

This hypothesis must be checked for every newly introduced native extension.
Do not attribute the incident to a new GDN kernel solely from this stack. Use
an otherwise identical `extension=off/on` A/B run. If this log predates the
extension, it cannot be evidence against that extension.

### H4: Native runtime or multiprocess interaction

A CoreX driver/runtime fault, custom IPC issue, or worker teardown race could
cause several TP ranks to fail together. This is lower priority until a
request-level reproducer has excluded H1-H3.

## Correlated LongBench V2 evidence

The evaluator emitted the following aggregate warning approximately 61 seconds
after the segmentation fault:

```text
2026-07-14 06:31:12 longbench_v2:
463 requests failed during inference
items=503, samples=503, correct_samples=19
avg@1=0.0378, pass@1=0.0378
elapsed=19450.9s
```

The sampled failures were HTTP 400 responses from the OpenAI-compatible chat
endpoint. Each request reserved 16,384 completion tokens and exceeded the
configured 100,000-token model limit:

| Message tokens | Requested completion | Total | Excess |
|---:|---:|---:|---:|
| 84,868 | 16,384 | 101,252 | 1,252 |
| 85,885 | 16,384 | 102,269 | 2,269 |
| 86,725 | 16,384 | 103,109 | 3,109 |

With the immutable evaluator setting `max_model_len=100000`, a request that
reserves 16,384 completion tokens can contain at most:

```text
100000 - 16384 = 83616 message tokens
```

At least the sampled requests exceeded that admission boundary. The reported
463 failures account for approximately 92.05% of the 503-item dataset, while
19 correct samples account for 3.78%.

### Relationship to the segmentation fault

The HTTP 400 responses are not a direct cause of the native crash. Context
length validation happens in the API layer before model execution, so a
rejected request does not launch paged attention or another GPU kernel.

The two observations can nevertheless belong to the same evaluation run and
share an indirect workload relationship:

```text
over-limit requests
  -> rejected by API validation, no GPU execution

admitted long-context requests
  -> model prefill and long-context paged decode
  -> possible trigger for the observed native decode-path fault
```

If most of the 463 failures were context-limit 400 responses, roughly 40
requests remained available for actual inference. Those admitted requests may
still have been close to the 83,616-token boundary and would heavily exercise
KV block tables, prefix caching, slot mapping, and paged-attention decode. The
61-second proximity between the service fault and evaluator summary is
consistent with a worker crash near the end of the run, but it is not proof of
causality.

Before drawing a stronger conclusion, group all 463 errors by type. In
particular, distinguish context-limit HTTP 400 responses from connection
resets, timeouts, HTTP 5xx responses, and worker-loss errors after the SIGSEGV.

### Optimization candidate: cap reserved completion tokens

LongBench V2 appears to reserve 16,384 completion tokens even though many
answers are expected to be much shorter. Instead of rejecting every request
whose `prompt_tokens + max_tokens` exceeds 100,000, an API compatibility layer
could evaluate the following exact cap:

```text
available_output_tokens = max_model_len - prompt_tokens
effective_max_tokens = min(request.max_tokens, available_output_tokens)
```

For example, an 84,868-token message could be admitted with at most 15,132
completion tokens instead of being rejected solely because it requested
16,384. This does not increase the model context window or permit more than
100,000 total tokens.

This is an experiment proposal, not an approved production change. It must be
implemented independently from kernel work and pass the following gates:

1. Requests whose prompt alone is at least 100,000 tokens still fail clearly.
2. The cap never becomes zero or negative for an admitted request.
3. Requests already within the limit remain byte-for-byte unchanged.
4. The response preserves valid `finish_reason`, usage, and streaming behavior.
5. Short-answer LongBench outputs match an uncapped reference whenever neither
   run reaches the effective completion limit.
6. Tool calls, reasoning content, JSON schema, and multimodal requests remain
   valid after request normalization.
7. The evaluator contract permits server-side normalization of `max_tokens`.
8. Success-rate gain is reported separately from model-quality and performance
   gains; a truncated output is not counted as an equivalent answer.
9. Cold/warm prefix-cache and long-context boundary tests remain crash-free.

Because request success rate has a 99% competition target, resolving this
admission mismatch may have a substantially larger score impact than a small
decode-kernel speedup. However, silent truncation or behavior changes are not
acceptable merely to turn HTTP 400 responses into HTTP 200 responses.

### Evidence to correlate the two events

For the request active immediately before `2026-07-14 06:30:11`, retain:

- request ID and evaluator item ID;
- message tokens and effective completion-token limit;
- whether admission performed a cap;
- cold/warm prefix-cache state and cached-token count;
- physical block count and block-table length;
- first decode-token time and last completed decode token;
- worker PID, TP rank, and GPU;
- final HTTP status or transport error observed by the evaluator.

If the crashing request was an admitted near-boundary request, prioritize H1
and the long-context reproduction matrix. If every admitted long request
completes and the crash belongs to another workload or service instance, keep
the LongBench admission problem and native crash as separate incidents.

## Required evidence for the next occurrence

Capture at least 100 log lines before the first fatal signal and record:

- repository commit and installed runtime file hashes;
- container/image identifier;
- worker PID, local rank, and GPU index;
- request ID and whether it was cold or prefix-cache warm;
- prompt length, context length, decode token index, and requested max tokens;
- computed and physical block counts;
- block-table length, slot mapping, and cached-token count;
- tensor shapes, dtypes, devices, strides, and contiguity at decode dispatch;
- the most recent successfully completed kernel stage;
- free HBM and process RSS immediately before the request.

## Reproduction matrix

Keep the request payload, model, seed, fixed evaluator arguments, and code SHA
constant. Change one diagnostic factor per run.

| Test | Prefix cache | New native extension | Context | Purpose |
|---|---:|---:|---:|---|
| R1 | on | off | failing length | Reference reproduction |
| R2 | on | on | failing length | Detect extension correlation |
| R3 | off | off | failing length | Isolate prefix-cache metadata |
| R4 | on | off | boundary - 1 | Detect block-boundary defect |
| R5 | on | off | boundary | Detect block-boundary defect |
| R6 | on | off | boundary + 1 | Detect block-boundary defect |
| R7 | on | off | short context | Confirm general decode health |

Useful boundary lengths include `15/16/17`, `31/32/33`, and the actual KV
cache block-size boundaries used by the runtime. Also test the same prompt as a
cold request followed by an identical warm request.

Diagnostic-only environment:

```bash
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
```

`CUDA_LAUNCH_BLOCKING=1` changes timing and must not be used for performance
results. It is intended to move the reported failure closer to the kernel that
actually caused it.

If the environment permits native dumps:

```bash
ulimit -c unlimited
```

Use the resulting core with the available native debugger to obtain C/C++ and
runtime frames. `TORCH_SHOW_CPP_STACKTRACES=1` can help checked Torch failures,
but may provide limited information for a raw SIGSEGV.

## Optimization safety gates

Any optimization touching GDN, attention, KV cache, block metadata, custom
IPC, tensor layout, or native kernels must pass all of the following before it
is considered a winner:

1. Four-card CUDA and collective preflight.
2. Actual-shape output/state parity against the reference path.
3. Input immutability checks for every tensor not documented as in-place.
4. Cold and warm prefix-cache requests with identical output hashes.
5. Context-boundary tests around physical block sizes.
6. At least 1,000 consecutive decode steps with no non-finite values.
7. Full API smoke, including image, tool-call, and long-prefix cases.
8. Repeated service starts and fatal-log scans.
9. Alternating baseline/candidate runs with the same request sequence.
10. Zero SIGSEGV, illegal-memory-access, CUDA/CoreX fatal, or worker-loss events.

An optimization is rejected immediately if it produces any native crash, even
when its throughput is higher. Allclose microbench results alone are not proof
of service safety because state drift or memory corruption can emerge only
after long decode or cache reuse.

## Decision tree

```text
Reproduces only with prefix cache warm
  -> audit block table, cached length, slot mapping, and recycle lifecycle

Reproduces with extension on but not off
  -> audit extension bounds, stream, device, aliasing, and launch error checks

Reproduces at one context/block boundary in both modes
  -> audit paged-attention metadata and vendor shape support

Reproduces randomly across unrelated shapes
  -> collect core dump and investigate CoreX runtime, IPC, and prior kernels

Does not reproduce
  -> retain as an unresolved incident; do not claim a root cause
```

## Current disposition

The incident is confirmed as a native decode-path crash. The precise corrupting
operation remains unproven from the supplied log alone. Future work should use
the reproduction matrix and evidence checklist above before changing kernels or
cache behavior in response to this incident.

## Follow-up verification on the winner baseline

The remote winner baseline was restored to commit
`b1d95009d52135a5b00bbac1c5ccc682c4539644` before reproduction. The repository
and installed runtime hashes match for both `paged_attn.py` and `qwen3_5.py`.
The installed `model_runner.py` also contains the chunked-prefill fix that clears
`prefix_cache_hit` after execution has passed the original cached region, so the
known computed-block-only table bug is not present in this baseline.

The reported `paged_attn.py:277` frame maps to:

```python
actual_max = int(seq_lens.max().item())
```

This operation synchronizes device state. It strengthens H3: the frame can be
where an earlier asynchronous cache-write, attention, GDN, linear, or collective
failure becomes visible. It does not identify `paged_attention_v1` as the first
faulting kernel. The current dispatch is:

- context length `<= 32768`: native paged-attention V1;
- context length `> 32768`: pure-PyTorch decode fallback;
- paged-attention V2: unavailable in the current BI100 runtime.

The diagnostic branch adds always-on host-side shape/capacity guards before
decode dispatch. Expensive physical block-ID checks, sparse metadata logs, and
post-cache-write device synchronization remain opt-in through
`BI100_PAGED_ATTN_DIAGNOSTICS=1`, so the default scoring path gains no extra GPU
synchronization.

### Baseline reproduction result

The unchanged winner runtime completed the first controlled TP=4 reproduction:

| Case | Result | Cached tokens | Output hash |
| --- | --- | ---: | --- |
| Prefix boundary, partial | pass | 8,176 | `fb7f24a13a246fd59ec6a661553b81fde7b1d5516c57cf730512c1942065dd59` |
| Prefix boundary, warm | pass | 11,600 | same as partial |
| 99,500-token prompt, cold | pass, 159.075 s | 0 | `a3dc73d02269b1b3682ed84197c3d2d0ddc39dfdb544f73fb3ea832f1fb30b4d` |
| 99,500-token prompt, warm | pass, 18.403 s | 99,296 | same as cold |

The service remained healthy and its log contained no SIGSEGV, fatal Python
error, OOM, worker loss, or traceback. The runtime reported 16,871 GPU blocks
and 6,553 CPU blocks. This run does not reproduce the incident, so it neither
proves a root cause nor closes the native-crash investigation. The next run
enables metadata range checks and synchronization immediately after
`reshape_and_cache`, then covers the 32,768/32,769 dispatch boundary.

The commit timeline also keeps H3 open. E-GDN-01 merged six GDN input
projections into one `MergedColumnParallelLinear` at `05:29 UTC`; the crash was
reported around `06:30 UTC`, and qualification documentation was committed at
`06:59 UTC`. The second fault dump was inside ixformer linear. This is only a
temporal correlation, but future reproduction must include an E-GDN-01 off/on
A/B or an equivalent parent/winner comparison before ruling out an earlier
linear-kernel fault that surfaced at the line-277 synchronization point.

### 256K readiness constraint

The submission configuration is now approved to use the model-native
`max_model_len=262144`; the earlier 100,000-token admission limit remains only
as historical evidence for the failed evaluation run.

The model text configuration declares `max_position_embeddings=262144`. With
the current 16-token KV block size and a measured 16,871 GPU blocks, the runtime
has nominal capacity for 269,936 tokens. This is enough for a 235K-token input
plus a 16K-token completion, but only leaves about 8K tokens beyond a full
262,144-token model window. Startup and request tests must therefore distinguish:

1. `235K input + requested output <= configured max_model_len`;
2. prompt-only lengths at the configured boundary;
3. prompt plus requested output beyond the configured boundary;
4. physical KV capacity from the logical model limit.

The 256K test must also qualify the long-context PyTorch decode fallback for
peak HBM, output parity, and latency. Nominal KV capacity alone is not evidence
that the 256K service path satisfies the throughput target.

## 2026-07-15 diagnostic qualification

The guarded TP=4 runtime completed the 32,768/32,769 dispatch boundary, partial
and warm prefix-cache reuse, a 235,000-token cold/warm request, and 1,000 forced
decode tokens. The log contained both native V1 and pure-PyTorch decode paths
and no guard failure, synchronization failure, SIGSEGV, fatal Python error,
OOM, or worker loss. The service remained healthy.

This run did not reproduce the original native crash and therefore does not
establish its root cause. The host-visible guards and long-context fallback
remain defensive containment.

The qualification did identify an independent long-prompt cache issue. The
GDN recurrent-state cache retained only 16 staged checkpoints, while the
262,144-token window with 8,192-token chunks needs 32. A 235,000-token warm
request consequently reported zero cached tokens and took 510.317 seconds.
Commit `e1ba860` raises both scheduler and worker retention to 32. The repeated
hardware test reported 234,544 cached tokens and completed the warm request in
41.090 seconds with the same output hash as cold. This fix improves prefix
reuse; it is not evidence that GDN checkpoint eviction caused the reported
SIGSEGV.
