# TP2 staging validation (2026-07-24)

## Scope

This run used `prep/M1-48-on-M1-49-20260722@f2c4b6e` on
`ssh-92d14d83`. It validates the isolated runtime, two-rank communication,
M1-49 layer selection, and event-profiler plumbing. It is not a TP4
qualification, performance A/B, cache trace, or promotion result.

No full model service was started. `computility-run.yaml`, `main`, repository
visibility, and system site-packages were not changed.

## Capacity boundary

- The host has two idle Iluvatar BI-V100 GPUs with 34,057,748,480 bytes each.
- The 26 model weight files total 71,903,776,776 bytes.
- At the fixed `--gpu-memory-utilization 0.9`, TP2 exposes at most
  61,303,947,264 bytes before runtime, KV cache, and activation overhead.
- The weights alone exceed that limit by 10,599,829,512 bytes, or 9.872 GiB.

Starting the 35B service on this host would therefore be an expected OOM and
would not reproduce the TP4 layer shapes. The run stopped at bounded component
tests.

## Runtime identity repair

The first `f70f1e8` overlay correctly failed closed because its install report
compared a profile-only repository copy of `xformers.py` with the actual
post-SDPA-plus-profile runtime file. The runtime differed by the required
`_run_sdpa_fallback` method and dispatch branch, not by an unknown mutation.

`8bef3f4` makes the checked-in canonical file represent the complete final
artifact and makes `patch_xformers_profile.py` produce stable LF text with a
terminal newline and no trailing whitespace. The exact final SHA-256 is:

`2f35a9a1f4af7b4d1473a3b81737b0abe0a04c9a5e00fc2f5374873b6d07bddb`

A fresh remote install then passed with:

- install exit code 0 and `qualified=true`;
- source, installed, and active-runtime XFormers SHA-256 identical;
- runtime identity exit code 0 and no reasons;
- `system_site_packages_modified=false`;
- Transformers 4.55.3 and CoreX vLLM 0.6.3 resolved inside the overlay.

The qualified overlay is
`/root/competition-m1-48-tp2-runtime-f2c4b6e`.

## Bounded GPU checks

- TP2 NCCL all-reduce passed on ranks 0 and 1; both observed the expected sum
  `3.0`, with no timed-out rank.
- M1-49 selector smoke returned 40 KV layers for `legacy40` and 10 for
  `full_attention`.
- The full-attention ordinals were exactly
  `3,7,11,15,19,23,27,31,35,39`.
- Event-profiler smoke passed independently on GPU0 and GPU1. Each emitted two
  events and two matching counters; region times were finite and positive and
  flush completed.
- Submission preflight passed all 8 checks. Local regression passed 403 tests
  with 25 dependency skips.

## Capacity-policy diagnostic

The fixed 881-request synthetic proxy was replayed at the legacy 16,878-block
capacity and the estimated M1-49 67,512-block capacity:

- Capacity alone raised LRU hit rate by 10.244 percentage points.
- At estimated M1-49 capacity, frequency eviction exceeded LRU by another
  13.607 percentage points, adding 195,498 block hits.
- Capacity pressure remained, so the proxy does not support discarding the
  frequency-policy direction after M1-49.

This proxy has synthetic block identities, no GDN availability, and no measured
TP4 capacity, TTFT, throughput, or score. It cannot unlock M1-29 or authorize a
runtime policy. A complete same-session privacy-safe 881 trace is still
required.

## Decision

The runtime and measurement plumbing are ready for a healthy TP4 host. M1-49
still requires its fixed TP4 A/B and long-context gates. M1-48 remains locked
behind M1-49 and must use the fixed 235K profile protocol. No result from this
TP2 run is eligible for `main` or submission YAML.

Structured evidence:
`docs/experiments/evidence/TP2_STAGING_VALIDATION_20260724.json`.
