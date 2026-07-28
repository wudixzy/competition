# M1-97 legacy candidate retest audit

Date: 2026-07-28

## Selection rule

Policy v2 permits a bounded retest only when an older candidate:

- has meaningful measured hotspot or end-to-end benefit;
- was closed primarily by an oracle or continuation threshold that policy v2
  has independently corrected;
- has no separate fatal, OOM, capacity, protocol, capability, or model-output
  regression;
- can be rerun without scanning tiles, tolerances, chunk sizes, or YAML
  parameters.

A new policy does not erase historical results. Old reports retain their
declared decision; a retest creates a new result with a new identity.

## Retest queue

### P0: M1-91 compensated W13

- historical fixed/routed speedups: `6.094x` and `5.464x`;
- historical rejection: candidate-versus-vendor sequence relative L2 around
  `1.42e-5` to `1.45e-5`;
- no fatal, OOM, lifecycle, or fixed high-precision failure.

M1-96 has now completed the one permitted unchanged-kernel v2 rerun. The
candidate is closer than vendor to the high-precision reference on both seeds
and remains about `5.47x` faster on the W13 routed boundary.

M1-98 then integrated it into the complete production routed-MoE boundary.
The candidate improved aggregate error and mismatch count, but one seed's
maximum absolute error increased from `1.22e-4` to `1.83e-4`. More
importantly, it was only about `1.1%` faster than the current direct control.
That is below the fixed `5%` continuation threshold even if the isolated
worst-error rule were relaxed. M1-91/M1-98 is closed without TP4 evaluation
or a production binary change.

### P1: M1-47 fused paged prefill

- corrected production-shape core speedup: `2.553x` at 74K;
- historical TP4 cold-TTFT improvement: `3.910%` at 65K and `8.832%` at 235K;
- warm results did not regress;
- numerical and dispatcher gates passed.

M1-47 was closed because both cold cases were required to improve by 20%.
That threshold was too strong for a continuation screen and discarded a real
235K service improvement. Do not rerun its numerical microbenchmark or scan
the kernel. Run at least three fixed-order paired TP4 A/B repetitions from
the corrected artifact, evaluate the paired distribution, and require at
least 5% targeted end-to-end improvement without decode, warm, quality, or
capacity regression.

### P2: E-PREFIX-08 cold-chunk hybrid

- cold complete and partial boundaries improved `22.15%` and `22.06%`;
- warm boundary stayed exact and changed by only `0.88%`;
- maximum absolute errors were only `1.19e-7` and `2.38e-7`;
- relative L2 was `3.49e-5` to `3.55e-5`.

The old relative-L2-only rejection may be dominated by near-zero reference
values. Permit one unchanged-algorithm numerical rerun against a
high-precision attention oracle, with vendor-control non-inferiority and
fixed next-token follow-up if it passes. No merge formula, tile, threshold,
dtype, or tolerance scan is allowed.

### P3: M1-28 WMMA QK

- QK microbenchmark speedup: `1.608x`;
- all outputs finite and maximum absolute errors passed;
- two output relative-L2 cases were `1.3834e-5` and `1.8299e-5`.

This is a lower-priority, low-cost high-precision oracle check. It remains
closed if vendor FP32 BMM is more accurate, or if the primitive cannot show a
credible integrated service benefit. A microbenchmark pass alone cannot
restart the full paged-attention implementation.

## Do not rerun unchanged

| Candidate | Reason |
|---|---|
| M1-63 pairwise W13 | Fast, but less accurate and slower on the routed boundary than the now-qualified M1-91 algorithm. It is dominated. |
| M1-61 exact-W2 hybrid | The W2 copy is byte-exact and reusable, but the complete candidate embeds the rejected direct W13. Recombine the exact-W2 component only after M1-91 integration. |
| E-MOE-18 W13 matvec | The fastest useful arithmetic is superseded by M1-91; its FP64 variants were grossly wrong. |
| E-ATTN-06 direct paged decode | The 100K stress maximum error reached about `0.059`, not a small reduction-order discrepancy. |
| E-MOE-04 weighted reduce | A real 1,000-token output hash divergence was observed while complete-path gain was only about `1.05x`. |
| M1-55 attention variants | One path was slower than baseline; the other delivered only about `1.18x` and required 345-500 MiB more workspace. |
| M1-37 and E-PREFIX-06 | Large-row cases had sparse catastrophic errors, so the reference choice was not the cause. |
| M1-38 and M1-71 complete binaries | Both depend on the old direct W13. Architectural components may be recombined, but the old binaries should not be rerun. |

## Execution order

1. retain the completed M1-91/M1-98 evidence; do not rerun the integrated
   candidate;
2. run the corrected M1-47 three-pair
   TP4 A/B;
3. run one E-PREFIX-08 high-precision oracle;
4. run M1-28 only if capacity remains and no higher-value candidate has
   already supplied the needed attention gain.

No item in this audit authorizes a YAML, default-runtime, or `main` change.
