# M1-36 Direct GDN Restore Copy Probe - 2026-07-21

## Objective

M1-35 retained canonical recurrent states and brought `admission64/direct` to
within `57.7428` weighted points of the predeclared `+5%` cache-stage gate.
The remaining bounded cache micro-optimization was the worker restore path:
it first called `saved_state.to(device)` and then copied that temporary GPU
tensor into the live GDN buffers.

This experiment measures whether copying the CPU state directly into the live
GPU buffers saves enough absolute time to justify a runtime change and another
TP4 matrix. The gate was fixed before measurement at exact output plus at
least `15 ms` median saving per restore. That is approximately the per-request
saving needed to close the M1-35 score gap; a relative speedup without useful
absolute latency is not sufficient.

## Probe

`tests/bench_gdn_restore_copy.py` reproduces one rank's complete recurrent
state using the production shapes:

| State | Shape | Dtype |
| --- | --- | --- |
| convolution | `[30, 2048, 3]` | `float32` |
| temporal | `[30, 8, 128, 128]` | `float32` |

The two tensors total `16,465,920` bytes per rank. The benchmark alternates
the current temporary-GPU path and direct CPU-to-live `copy_` across seven
repeats, with 50 restores per repeat and device synchronization around every
timed batch. It first compares both destination tensors with `torch.equal`.

## CoreX Result

The probe ran on the four-card BI100 development instance using GPU 0. Its
artifact is
`/root/competition-m1-32-latest/bench_runs/m1_36/restore_copy_probe/result.json`.

| Metric | Result |
| --- | ---: |
| Exact destination state | true |
| Current path median | `1.7792 ms` |
| Direct path median | `1.5627 ms` |
| Absolute saving | `0.2165 ms` |
| Relative speedup | `1.1385x` |
| Required absolute saving | `15.0 ms` |

The benchmark deliberately returned `rc=1` because the candidate missed its
performance gate. Stderr contained only the existing `pynvml` deprecation
warning.

## Decision

Status: `PERFORMANCE_REJECTED`.

Do not integrate the direct-copy variant and do not run another fixed matrix.
Even eliminating the measured direct restore entirely would be far below the
latency needed to close the M1-35 stage gap. This exhausts the bounded GDN
state-transfer micro-optimization path; subsequent work must use full-trace
evidence to evaluate a structural cache policy or move to cold-prefill
attention work. Default `fine32/direct`, `computility-run.yaml`, and `main`
remain unchanged.
