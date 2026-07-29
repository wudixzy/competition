# M1-130 FP32 cuBLAS concurrency probe

Date: 2026-07-29

Status: fixed-shape BI100 execution completed. Independent FP32 QK/PV
submission on two streams regressed both shapes, so the double-buffer pipeline
route is closed.

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

## Result

Source commit `73da58f0ab253407fde7e460d783845c226f38fd` ran on
`ssh-73ca29ba`. Both cells executed successfully and produced bit-identical
sequential and concurrent QK/PV outputs:

| Case | QK only ms | PV only ms | Sequential ms | Concurrent ms | Speedup |
|---|---:|---:|---:|---:|---:|
| q8176 | 0.699 | 0.637 | 1.309 | 1.352 | 0.968x |
| q5616 | 0.491 | 0.481 | 0.948 | 0.964 | 0.983x |

The median speedup was `0.976x`, below the fixed `1.10x` threshold, and each
cell was below the `1.05x` floor. Concurrent submission increased latency by
about 3.3% and 1.7%, respectively. This shows that the installed BI100/CoreX
BLAS path does not expose useful overlap for these two dominant FP32 GEMMs.

All cell execution, scoped cleanup, fatal scan, timeout scan, pre/postflight,
and repeated four-GPU preflight checks passed. The runner's nonzero overall
status reflects the performance rejection only. M1-131 double buffering, TP4
service testing, runtime overlay changes, and `main` or YAML changes are not
authorized.
