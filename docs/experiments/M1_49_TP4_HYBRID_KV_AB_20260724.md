# M1-49 TP4 Hybrid KV A/B (2026-07-24)

## Scope

This fixed-order A/B compared `legacy40` with `full_attention` KV accounting
on four BI-V100 GPUs. The measured source revision was
`a24023c49300e14a80a02f601792e46418be70cb`. The runtime overlay was unchanged
between arms.

Both arms used the fixed evaluator contract: TP4, 262144 maximum model length,
8192 chunked prefill, concurrency one, `admission64/direct`, LRU eviction, CPU
KV offload disabled, and fused prefill disabled. The only arm difference was
`BI100_HYBRID_KV_ACCOUNTING`.

## Results

| Metric | `legacy40` | `full_attention` | Ratio or delta |
|---|---:|---:|---:|
| GPU blocks | 16,878 | 67,512 | 4.000000x |
| CPU blocks | 6,553 | 26,214 | 4.000305x |
| Immediate warm cached tokens | 65,520 | 65,520 | 0 |
| Immediate warm latency | 3.387843s | 3.033299s | 0.895348x |
| Post-pressure cached tokens | 16 | 65,520 | +65,504 |
| Post-pressure latency | 106.829777s | 2.935682s | 0.027480x |

The capacity gate required at least 3.5x and the immediate-warm latency gate
allowed at most 1.02x. Both passed. The two 135040-token pressure requests and
all target requests returned identical redacted message identities across
arms. All request, startup, fatal-scan, cleanup, and four-GPU preflight gates
passed.

The refreshed warm request was 3.330281s for the candidate versus 3.027521s
for control, a 1.100002x single-sample regression. This field is not the frozen
latency gate, and one eight-token request is insufficient to claim a steady
regression. It remains a residual risk for the dataset replay; it is not
silently discarded.

## Cleanup Validation

Each service lifetime left one PID-1-owned zombie in its process group. The
new cleanup logic reported `zombie_count=1`, confirmed that no live group
member or bound port remained, and continued. Both service arms completed,
and final `cleanup.rc`, `comparison.rc`, `preflight_comparison_final.rc`, and
`overall.rc` were zero.

## Decision

M1-49 passes its capacity and pressure A/B and is admitted to the frozen
131K/235K/262K correctness gates. This result does not qualify long-context
output equivalence, the selected dataset, final weighted score, or a submission
default. `computility-run.yaml` remains unchanged until all downstream gates
pass.

Structured evidence:
`docs/experiments/evidence/M1_49_TP4_HYBRID_KV_AB_20260724.json`.
