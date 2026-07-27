# M1-61 Exact-W2 Hybrid MoE

## Decision

`REJECT FOR PRODUCTION`.

M1-61 tested whether the M1-60 numerical failure could be isolated to the
direct W2 kernel. The candidate retained direct W13, copied only the selected
W2 experts, and restored the existing vendor BMM plus serial-float routed
reduction.

The W2-only copy was byte-exact and the hybrid was materially faster, but a
new fixed seed exposed direct-W13 error above the `1e-5` hard limit. The
hybrid endpoint and 500-step sequence therefore also failed. Do not add a
production dispatch, modify `computility-run.yaml`, or run a TP4 service gate
for this candidate.

## Lineage

```text
branch  exp/M1-61-exact-w2-hybrid-20260727
probe   0382bb3
gate    c039504
host    ssh-73ca29ba
device  physical GPU3, isolated as cuda:0
```

The fixed rank-local shape was:

```text
experts=256 top_k=8 hidden=2048 intermediate=128 dtype=float16
```

The test used nine repeats of 300 iterations after 30 warmups and a 500-step
fixed-seed sequence. Raw generated tensors and model data were not recorded.
Structured evidence is under
[`evidence/M1_61_EXACT_W2_HYBRID`](evidence/M1_61_EXACT_W2_HYBRID).

## Result

| Boundary | Result | Gate |
| --- | ---: | ---: |
| Selected W2 copy | byte-exact | exact |
| Direct W13 relative L2 | 2.452e-5 | <= 1e-5 |
| Hybrid relative L2 | 9.404e-5 | <= 1e-5 |
| 500-step relative L2 | 7.526e-5 | <= 1e-5 |
| Maximum step relative L2 | 3.204e-4 | <= 1e-5 |
| Finite steps | 500/500 | 500/500 |
| Exact steps | 59/500 | diagnostic only |

The fixed and routed paths were faster:

| Path | Baseline | Hybrid | Speedup | Gate |
| --- | ---: | ---: | ---: | ---: |
| Fixed expert path | 0.26573 ms | 0.10923 ms | 2.433x | >= 1.5x |
| Routing plus expert path | 0.32731 ms | 0.16979 ms | 1.928x | >= 1.25x |

Selected W13 plus W2 copying took `0.06375 ms`; W2-only copying took
`0.02391 ms`. The data-plane hypothesis was valid, but the direct GEMV
reduction order was not numerically robust.

## Interpretation

M1-60 measured direct W13 at `7.302e-6` with seed `20260716`. M1-61 used seed
`20260727` and measured `2.452e-5`. A single favorable tensor was therefore
insufficient evidence for the direct W13 path. The remaining drift is not
caused by selected-weight copying or the routed reduction; it originates at
the custom W13 matvec boundary and propagates through activation and W2.

Meeting the hard limit would require reproducing the vendor GEMM numerical
boundary, not adjusting a launch grid or accepting a looser tolerance.
Earlier pointer-batched FP32 cuBLAS work was exact but slower than the current
gather path. Per the stop rule, no tile, warp, or tolerance scan follows.

## Next Step

1. Keep M1-61 as negative evidence and do not integrate its dispatch.
2. Establish a full-model TP4 reference run with the numerically rejected
   direct MoE and packed GDN paths disabled.
3. Profile that quality-safe service before selecting another kernel target.
4. Prefer optimizations that preserve vendor GEMM boundaries or operate on
   byte-exact data movement, cache scheduling, and prefill paths.
