# TP4 candidate stack readiness

## Branch

```text
integration/tp4-candidates-20260715
head before this note: b6431c0
base: integration/perf-winners@adab4bc
```

This branch combines production code for four independently qualified
single-card candidates without changing `computility-run.yaml`:

| Candidate | Branch source | Measured primitive/full-boundary saving |
| --- | --- | ---: |
| E-ATTN-03 packed local QGKV | `5bebe8c` | ~0.24 ms/token projected |
| E-GDN-03 fused causal conv | `3a1a458`, `6d7edff` | ~1.35 ms/token projected |
| E-GDN-05 gated norm output | `d823dbd` | ~1.63 ms/token projected |
| E-MOE-11 combined exact MoE tail | `d6ac803` + E-MOE-11 | ~1.83 ms/token projected |

The unqualified additive projection is approximately `5.1 ms/token`, or
roughly 7% against the current 13.3-13.5 Output TPS range. This would imply
about 14.3-14.5 TPS, still below the 20 TPS competition target. Treat this only
as prioritization; shared launch and memory effects require service A/B.

## Build and static gates

`patch_ops.sh` builds all three CoreX extensions into the discovered vLLM
package before installing the model file:

```text
corex_gdn_causal_conv.so
corex_gdn_gated_norm.so
corex_moe_exact_reduce.so
```

Combined local validation passes:

```text
Python compile: pass
shell syntax: pass
git diff check: pass
P0 + attention/GDN/MoE units: 56 run, 13 GPU-only skipped, 0 failed
```

## Qualification order

Do not benchmark all candidates first. On a healthy four-card host:

1. Start the qualified E-MOE-03/E-GDN-01 baseline and capture full smoke,
   fixed benchmark, 1,000-token hash, and cold/warm long-context output.
2. Qualify E-ATTN-03 alone.
3. Enable E-GDN-03 with `BI100_GDN_COREX_CAUSAL_CONV=1`; compare against its
   explicit `0` fallback.
4. Enable E-GDN-05 with `BI100_GDN_COREX_GATED_NORM=1`; compare against `0`.
5. Enable E-MOE-11 in two stages: first compare
   `BI100_MOE_COREX_EXACT_REDUCE=1/0`, then compare
   `BI100_MOE_FUSED_ACTIVATION=1/0`.
6. Run the all-enabled stack only after every individual 1,000-token hash is
   identical. Repeat three interleaved service A/B pairs.
7. Finish with full smoke, 99.5K and 235K/256K cold-warm equality, cache-hit
   accounting, and log scans for non-finite/OOM/worker loss/native crashes.

E-ATTN-03 has no environment fallback because it changes parameter layout;
use the qualified integration model file for its baseline. The three CoreX
extensions and the MoE activation fusion are independently switchable. CoreX
extensions fail closed on unsupported dtypes or shapes.

## Current blocker

Physical GPU0 on `ssh-a2d0a302.default.gpu.phanthy.com` remains unusable while
GPU1-3 pass single-card tests. TP2 model loading fails while allocating routed
MoE `w13_weight`, with only 14 MiB free; `cpu_offload_gb=8` does not alter that
initialization peak. A healthy TP4 host or platform reset is required for the
service gates above.
