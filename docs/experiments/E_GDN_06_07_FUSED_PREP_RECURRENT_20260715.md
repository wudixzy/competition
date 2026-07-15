# E-GDN-06+07: Fused decode preparation and recurrent update

> Superseded shape note: these probes followed stale 48-value-head source
> comments and used 12 local heads. E-GDN-09 audited the exact recurrent
> candidate at the checkpoint's real eight local heads and still found a
> regression. The nonexact fused-prep variants remain rejected.

## Scope

E-GDN-07 v1 showed that a fused recurrent kernel is numerically stable but
only 1.28-1.31x faster in repeatable serial runs. This experiment expands the
operation boundary to include q/k head expansion, FP16 L2 normalization, FP32
conversion, and recurrent state update.

```text
base: 74865b9
FP32 normalization: ffa3e33
PyTorch FP16 normalization: 5c8978f
half-semantics normalization: 60d1def
PyTorch inverse scalar: 37e689e
```

All variants use the real TP-rank decode shapes: four local q/k heads, twelve
local value heads, dimension 128, and FP32 temporal state. Every extension
compiled successfully with CoreX Clang and ran on physical GPU1.

## Results

| Variant | Speedup | Random output max abs | Random state max abs | Close | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| FP32 normalization inside kernel | 2.179x | 3.05e-5 | 3.80e-4 | no | reject |
| PyTorch FP16 norm before mapped kernel | 0.992x | 1.49e-8 | 1.79e-7 | yes | reject |
| half multiply/reduction/rsqrt inside kernel | 2.318x | 2.54e-5 | 5.17e-4 | no | reject |
| PyTorch inverse scalar, final multiply in kernel | 0.961x | 2.04e-5 | 3.43e-4 | no | reject |

The fast FP32 and half variants change normalization/reduction semantics enough
to fail the 1,000-random-token state and output gate. Preserving the complete
PyTorch FP16 normalization restores the E-GDN-07 error bounds but also retains
its launch overhead, making the fused candidate slightly slower. Passing only
the PyTorch inverse scalar does not reproduce the final FP16 multiply closely
enough on the CoreX kernel and also has no performance benefit.

The best numerically valid comparison was:

| Path | Median (ms) | P10 (ms) | P90 (ms) |
| --- | ---: | ---: | ---: |
| Reference full prep + recurrent | 0.211703 | 0.211585 | 0.212190 |
| PyTorch FP16 norm + mapped kernel | 0.213514 | 0.213425 | 0.213592 |

Remote artifacts are untracked under:

```text
/root/competition/bench_runs/20260715_E_GDN_06_07/
```

The benchmark exits 1 when a compiled candidate fails numerical close; this is
not a compiler failure. Build logs for all four variants contain no error.

## Decision

`REJECT AS PERFORMANCE WINNER`. None of the tested boundaries satisfies both
the random-sequence numerical gate and the 1.5x full-boundary performance gate.
Do not integrate these kernels into the model and do not relax the state/output
tolerance. Further work should move to a different hotspot or use an organizer-
supplied operator with matching FP16 reduction semantics.
