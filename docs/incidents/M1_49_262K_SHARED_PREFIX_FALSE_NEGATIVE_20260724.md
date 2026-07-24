# M1-49 262K Shared-Prefix Gate False Negative (2026-07-24)

## Incident

The fixed M1-49 TP4 long-context run completed both 262000-token inference
requests, but `long_262k_api.rc` and `overall.rc` were one. The client gate
required the first request to report exactly zero cached tokens. In the
sequential 131K, 235K, then 262K service lifetime, the first 262K request
legitimately reused 32 tokens from an earlier long-context request.

The persisted privacy-safe summary showed:

| Request | Prompt | Cached | Completion | Elapsed | Message identity |
|---|---:|---:|---:|---:|---|
| first | 262000 | 32 | 16 | 657.395s | `fc19a136...cef02c` |
| second | 262000 | 261984 | 16 | 9.013s | `fc19a136...cef02c` |

Both requests finished normally, their completion counts, finish reasons, and
full SHA-256 message identities matched, and service cleanup returned zero.
The server log contained no CUDA, OOM, process death, Gloo, assertion, or
segmentation-fault signature. This was a client contract false negative, not a
262K model-capacity failure.

## Correction

The long-context contract now carries `max_first_cached_tokens`. The 131K and
235K cold gates remain fixed at zero. The sequential 262K capacity gate permits
at most 32 tokens, exactly two complete 16-token cache blocks. It still
requires the second request to cache at least 261984 tokens and requires exact
output equivalence. A value of 33 or a weakened contract fails closed.

The 32-token allowance is fixed in the internal test harness. It is not exposed
in `computility-run.yaml`, is not used to improve cold-latency accounting, and
does not relax the final competition thresholds.

## Recovery

`M1_49_LONG_RESUME_FROM` enables a bounded recovery path in
`scripts/run_m1_49_long_context_gates.sh`. It accepts only a source run whose
overall and 262K API gates failed while cleanup, startup, runtime contract,
smoke, multimodal isolation, 131K, and 235K gates all passed. It also compares
the inherited startup invariant with the qualified M1-49 A/B candidate and
rejects any fatal server signature.

The recovery run starts a fresh fixed `full_attention/admission64/direct` TP4
service, requalifies the inherited 131K and 235K redacted summaries, reruns only
262K, performs before/after four-GPU preflights, and builds a new final
qualification. The original failed evidence remains immutable.

