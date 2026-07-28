# M1-107 main candidate

Date: 2026-07-28

Release base: `091c5bd07a317e96a8c6d5593f6c273e3b30b987`

This release promotes the most stable completed TP4 cache candidate for the
next official measurement. The submission environment now selects:

- `BI100_HYBRID_KV_ACCOUNTING=full_attention`
- `BI100_GDN_CACHE_POLICY=admission64`
- `BI100_GDN_RESTORE_MODE=hybrid64`

The fused prefill selector remains disabled because M1-108 did not have a
complete candidate measurement at release time.

M1-107 completed 18 requests in both arms. All nine cold/warm pairs were exact
within each arm and all 18 candidate/control output hashes matched. Relative
to the fixed control, effective cache hit rate increased from 49.93% to
62.78%, TTFT P90 improved from 17.290 seconds to 16.690 seconds, and Output TPS
P10 remained above the required floor at 21.986.

This is a stable measurement candidate, not a claim that the final official
targets are already met. Its focused weighted proxy was 7661.187 and its TTFT
P90 remained above five seconds. The official result must still determine
whether further optimization is required.
