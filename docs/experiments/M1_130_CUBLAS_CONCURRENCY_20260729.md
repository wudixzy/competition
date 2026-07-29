# M1-130 FP32 cuBLAS concurrency probe

Date: 2026-07-29

Status: implementation ready for fixed-shape BI100 execution.

## Rationale

The qualified M1-109 component spends most of its long-context time in
separate FP32 QK and PV GEMMs. M1-128 and M1-129 showed that FP16 GEMM inputs
improve component time, but both exceeded the fixed output relative-L2 gate.
That route is closed.

M1-130 keeps the exact FP32 arithmetic and asks a narrower question: can two
independent production-shaped QK and PV GEMMs make useful concurrent progress
on two BI100 streams? A positive result would justify implementing an exact
FP32 double-buffer pipeline that overlaps the next group's gather/QK with the
current group's softmax/PV/merge. A negative result avoids a complex pipeline
whose dominant GEMMs would serialize on the device.

The probe uses `torch.bmm` as a diagnostic-only way to exercise the installed
CoreX BLAS path. It is not evidence that the exact extension calls will overlap.
Timing uses synchronized host wall-clock intervals, so it includes submission
and synchronization overhead and is not presented as pure kernel duration.
It does not modify the runtime overlay, model, tokenizer, request semantics,
production source, or fallback. It does not modify `computility-run.yaml`.

## Fixed contract

Two cells run on separate physical GPUs:

- four heads, head dimension 256, and 512 key tokens;
- query lengths 8176 and 5616;
- FP32 inputs, accumulation, and outputs;
- five warmups and twenty measured trials;
- preallocated outputs so allocation is outside the timing window;
- fixed seed 20260729;
- TF32 disabled where the runtime exposes that switch.

Each cell compares the same independent QK and PV operations when submitted
serially and on two CUDA/CoreX streams. Sequential and concurrent outputs must
be finite, have relative-L2 at most `1e-7`, and maximum absolute error at most
`1e-5`.

The full probe qualifies only when:

- neither cell is below `1.05x` serial-over-concurrent speedup;
- the median of both cell speedups is at least `1.10x`;
- all numerical, lifecycle, fatal, timeout, postflight, and repeated GPU
  preflight checks pass.

Qualification only authorizes a separate M1-131 FP32 double-buffer component
implementation. It does not authorize TP4 service testing by itself, does not
authorize a runtime overlay, and does not authorize `main` or YAML changes.

## Stop rule

If the fixed probe fails, stream-overlapped QK/PV is closed without changing
the model path. No stream-count, launch-order, tile, threshold, or YAML scan is
allowed. The next investigation must return to measured end-to-end profile
data.
