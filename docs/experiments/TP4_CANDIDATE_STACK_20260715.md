# TP4 candidate stack readiness

## Branch

```text
integration/tp4-candidates-20260715
head before E-ATTN-04: 8e721cf
base: integration/perf-winners@adab4bc
```

This branch combines production code for five independently qualified
single-card candidates without changing `computility-run.yaml`:

| Candidate | Branch source | Measured primitive/full-boundary saving |
| --- | --- | ---: |
| E-ATTN-03 packed local QGKV | `5bebe8c` | ~0.24 ms/token projected |
| E-ATTN-05 exact paged K/V gather | E-ATTN-04 + E-ATTN-05 | ~35.9 ms/token at 64K; ~94.0 ms/token at 100K |
| E-GDN-03 fused causal conv | `3a1a458`, `6d7edff` | ~1.35 ms/token projected |
| E-GDN-05 gated norm output | `d823dbd` | ~1.63 ms/token projected |
| E-MOE-11 combined exact MoE tail | `d6ac803` + E-MOE-11 | ~1.83 ms/token projected |

The unqualified additive projection is approximately `5.1 ms/token`, or
roughly 7% against the current 13.3-13.5 Output TPS range. This would imply
about 14.3-14.5 TPS, still below the 20 TPS competition target. E-ATTN-05 is
not included in that generic projection because it activates only above 32K;
its context-dependent saving is listed separately in the table. Treat all
projections only as prioritization; shared launch and memory effects require
service A/B.

E-ATTN-06 tested a direct split-K paged decode and was faster than E-ATTN-05,
but failed the numerical gate at 100K (17/100 requests exceeded `1e-3`, worst
absolute error 0.05937). It is not part of this stack.

## Build and static gates

`patch_ops.sh` builds all four CoreX extensions into the discovered vLLM
package before installing the model file:

```text
corex_gdn_causal_conv.so
corex_gdn_gated_norm.so
corex_moe_exact_reduce.so
corex_paged_kv_gather.so
```

Combined local validation passes:

```text
Python compile: pass
shell syntax: pass
git diff check: pass
P0 + attention/GDN/MoE units: 79 run, 21 GPU-only skipped, 0 failed
```

## Qualification order

Do not benchmark all candidates first. On a healthy four-card host:

1. Start the qualified E-MOE-03/E-GDN-01 baseline and capture full smoke,
   fixed benchmark, 1,000-token hash, and cold/warm long-context output.
2. Qualify E-ATTN-03 alone.
3. Qualify E-ATTN-05 at 32K-1/32K/32K+1, 64K, 96K, and 100K with
   `BI100_ATTN_COREX_PAGED_GATHER=1/0`, including cold/warm requests.
4. Enable E-GDN-03 with `BI100_GDN_COREX_CAUSAL_CONV=1`; compare against its
   explicit `0` fallback.
5. Enable E-GDN-05 with `BI100_GDN_COREX_GATED_NORM=1`; compare against `0`.
6. Enable E-MOE-11 in two stages: first compare
   `BI100_MOE_COREX_EXACT_REDUCE=1/0`, then compare
   `BI100_MOE_FUSED_ACTIVATION=1/0`.
7. Run the all-enabled stack only after every individual 1,000-token hash is
   identical. Repeat three interleaved service A/B pairs.
8. Finish with full smoke, 99.5K and 235K/256K cold-warm equality, cache-hit
   accounting, and log scans for non-finite/OOM/worker loss/native crashes.

E-ATTN-03 has no environment fallback because it changes parameter layout;
use the qualified integration model file for its baseline. The four CoreX
extensions and the MoE activation fusion are independently switchable. CoreX
extensions fail closed on unsupported dtypes or shapes.

## Current blocker

Physical GPU0 on `ssh-a2d0a302.default.gpu.phanthy.com` remains unusable while
GPU1-3 pass single-card tests. TP2 model loading fails while allocating routed
MoE `w13_weight`, with only 14 MiB free; `cpu_offload_gb=8` does not alter that
initialization peak. A healthy TP4 host or platform reset is required for the
service gates above.
