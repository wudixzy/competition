# E-MOE-17: Contiguous W13 cuBLAS algorithm scan

## Hypothesis

E-MOE-16 showed that the gathered contiguous W13 linear is 39.4% of the
current routed boundary. E-MOE-07 rejected eight pointer-batched GEMVs, but did
not establish whether another cuBLAS algorithm is faster for the single
contiguous `2048x2048` W13 matrix used by E-MOE-13.

E-MOE-17 directly invoked CoreX cuBLAS `GemmEx` algorithms for the same FP16
input, weight, FP32 accumulation, and FP16 output contract as `F.linear`. It
also tested `Hgemm`. The CoreX 3.2.3 header does not expose `GemvEx`, so that
unsupported API was removed after a compile-time capability check.

## Method

`tests/bench_moe_w13_cublas.py` scanned default algorithms `-1,0..15` and
tensor-op modes `99..107` on GPU1. Each case used 20 warmups and 7 repeats of
300 iterations. Exact modes were also measured through the complete fixed and
routed E-MOE-13 boundary, followed by 100 random routed exactness steps.

## Results

Baseline:

```text
F.linear W13: 0.136872 ms
fixed full:   0.265920 ms
routed full:  0.330790 ms
```

All `GemmEx` modes produced bit-exact W13 and final outputs. The fastest naked
W13 result was mode 5 at 0.131065 ms (`1.0443x`), but its fixed full result was
only `1.0023x`. The best fixed full mode was 102 at `1.0041x`; its routed
result was:

```text
baseline routed:  0.330790 ms
candidate routed: 0.330435 ms
speedup:          1.0011x
sequence exact:   100/100, max_abs=0
```

`Hgemm` was both slower (`0.4870 ms` W13) and non-exact (`max_abs=0.001602`).

Raw artifact:

```text
/root/competition/bench_runs/20260715_E_MOE_17/gpu1-scan.json
```

## Decision

`REJECT FOR PRODUCTION`. CoreX/PyTorch already selects an effectively optimal
cuBLAS path for the complete boundary. Naked operator differences do not
survive composition, and the routed gain is far below 5%. Further W13 work
must use a genuinely fused or shape-specific kernel rather than another
cuBLAS algorithm wrapper.
