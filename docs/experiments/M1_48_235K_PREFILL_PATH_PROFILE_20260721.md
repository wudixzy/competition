# M1-48: 235K prefill path coverage profile

Status: measurement implementation complete; fixed TP4 runtime evidence pending.
This is a private diagnostic branch based on the qualified M1-45 cache stack at
`4340424`. It does not change model mathematics, Docker, `computility-run.yaml`,
`main`, or repository visibility.

## Why this profile is required

M1-11 measured `layer.full_attn` at 68.788% of a 235K cold model forward. Its
event filter did not include `paged_attn.*`, so that inclusive number did not
measure the fraction addressable by the later fused paged-prefix kernel.

M1-47 then accelerated its fixed 235K core case by 2.5770x, but improved the
235K cold service TTFT by only 8.832%. The candidate dispatched on all four
ranks and preserved numerical and request correctness, so more launch-grid or
tile scans are not justified. The missing evidence is production path
coverage: projection, norm/RoPE, dense attention, paged-prefix attention,
gating, output projection, and non-model service time must be separated.

## Instrumentation contract

The profiler remains disabled unless all diagnostic variables are supplied:

```text
BI100_PROFILE=1
BI100_PROFILE_MODE=event
BI100_PROFILE_INCLUDE_STARTUP=0
BI100_PROFILE_FILTER=model.*,layer.*,full_attn.*,paged_attn.*,moe.*,gdn_prefix.*
```

Event mode records CUDA start/end events and synchronizes once at the end of
each model forward. The profile contains:

- mutually exclusive model regions for embedding, norms, GDN, full attention,
  MoE, and final norm;
- full-attention subregions for QGKV projection, norm/RoPE, the attention call,
  gate, and output projection;
- nested `paged_attn.prefix_pytorch` time;
- per-forward context/query lengths and paged dispatch counts;
- worker-side model-to-flush time and the gap since the prior forward.

Only durations, bounded path names, integer geometry, booleans, and counts are
logged. Raw tokens, messages, images, model output, content hashes, GDN keys,
and physical KV block identities are forbidden.

## Fixed remote protocol

1. Use four BI100 GPUs, TP4, max model length 262144, block size 16,
   `max_num_batched_tokens=8192`, and `max_num_seqs=1`.
2. Use M1-45 `admission64/direct` and its qualified content CPU KV tier. Keep
   M1-46 block-major transfer and M1-47 fused attention disabled.
3. Start from an empty request cache and run `tests/long_context_api.py` with an
   exact 235000-token prompt, one output token, and the fixed deterministic
   seed. The cold request is the profile target; the warm request is retained
   only as a correctness check.
4. Summarize the service log with:

```text
python3 tests/summarize_prefill_path_profile.py SERVICE_LOG \
  --expected-prefill-tokens 235000 \
  --expected-processes 4 \
  --client-summary LONG_CONTEXT_SUMMARY \
  --candidate-core-speedup 2.5770191951430728 \
  --out PROFILE_SUMMARY
```

5. Archive source, runtime, service-log, request-summary, and profile-summary
   SHA-256 values. Stop the profile service after collection. Profiling data is
   never used as a formal TPS result.

## Profile gates

- exactly four processes and one request group totaling 235000 prefill tokens;
- identical forward metadata and dispatch counters on every TP rank;
- exclusive model regions close `model.forward` within 3%;
- full-attention subregions close `layer.full_attn` within 3%;
- paged-prefix time does not exceed its inclusive attention region;
- model-forward time closes 90% to 105% of client cold latency;
- no NaN/Inf, OOM, worker loss, collective error, timeout, or fatal log record.

The summarizer fails closed on incomplete records, rank drift, non-finite
timings, duplicate/missing target requests, or coverage outside these bounds.

## Decision rule

First compare the measured paged-prefix share with the M1-47 2.5770x core
speedup using Amdahl's law. If the projection explains the observed 8.832%
service gain, M1-47 remains closed and no fused-attention parameter scan is
reopened.

Only the largest remaining region with enough service-level headroom may
unlock a new implementation experiment. That experiment must predeclare one
primary design, at most one structural alternative, numerical parity, and a
235K cold service improvement of at least 20%. A core-only win cannot authorize
promotion. `computility-run.yaml`, `main`, 262K qualification, and the complete
881-request replay remain locked until a candidate passes its service gate.
