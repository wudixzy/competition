# M1-175 FP16-QK cross-instance reproduction

## Status

M1-175 reproduced the qualified M1-162 operator result on a second BI100
instance. The fixed 16K, 32K, and 64K cells passed the calibrated numerical
contract and retained 15.3%-17.2% speedups over M1-109. This strengthens the
operator evidence but does not authorize real-activation replay, TP4 service
testing, a production overlay, YAML changes, or `main` promotion.

## Environment

- source revision:
  `1e0410f080031a95edb7ec413ffba20f2a1a2608`;
- instance: `ssh-1f88d35a`;
- physical GPU 1: healthy `Iluvatar BI-V100`, 34,057,748,480 bytes free;
- physical GPU 0: excluded after the independent preflight timed out at
  `mem_get_info` and reaped its own process group;
- CoreX 3.2.3 and Torch 2.1.0;
- fixed FP16, head dimension 256, block size 16, GQA 4:1, and query length
  8,176.

Only the source, builders, benchmark cells, and frozen numeric contract were
transferred. Every transferred file was SHA-256 checked against the private
experiment branch before compilation. The current candidate source contains
the M1-173 batched-PV code only under an undefined compile-time macro; the
default M1-175 build retains the M1-162 execution path.

## Result

| Cell | Baseline ms | FP16-QK ms | Speedup | Historical speedup |
| --- | ---: | ---: | ---: | ---: |
| 16K | 72.073 | 62.501 | 1.1531x | 1.1536x |
| 32K | 136.049 | 116.704 | 1.1658x | 1.1652x |
| 64K | 263.856 | 225.103 | 1.1722x | 1.1707x |

The maximum absolute change in speedup relative to the original M1-162
instance was 0.123%. Candidate relative-L2 error remained within
`1.000008x` ordinary FP16 rounding error, maximum-absolute error remained
within `1.000733x`, and LSE relative L2 stayed below `3.67e-8`. All outputs
and LSE values were finite; candidate repeats were bit-exact.

The newly compiled artifacts have different byte hashes from the historical
instance. M1-175 therefore does not claim reproducible binaries. It binds the
source and contract hashes and establishes reproducible numerical and timing
behavior on the fixed tensors.

## Decision

Keep M1-162 as the mature prefill candidate. Cross-instance variance is now
too small to explain away its operator gain. Do not spend the one healthy GPU
on another synthetic shape sweep. The next required layer remains four-rank
real-activation replay followed by a short TP4 service A/B when a healthy
four-card instance is available.

GPU1 passed identical before/after allocation and 1024-square matrix
preflights with no free-memory loss. No run-owned process or GPU process
remained, and the fatal scan was empty. Privacy-safe evidence is under
`docs/experiments/evidence/M1_175_FP16_QK_CROSS_INSTANCE_20260804`.
