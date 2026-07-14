# E-GDN-07: Fused CoreX recurrent update

## Scope

E-CAP-02 proved that the image's CoreX Clang can build Torch CUDA extensions.
E-GDN-07 uses that path to fuse the decode recurrent state's decay,
`k @ state`, delta, rank-one state update, and `q @ state` into one kernel.
The reference path uses separate PyTorch pointwise, BMM, and `baddbmm_` calls.

```text
base:       9724b5f
prototype:  278a0d9
random test: 182a38b
integration candidate (rejected): d8b5cd4
```

## Correctness

The real per-rank shape is `(batch=1, heads=12, dim=128)` with an FP32
`(1, 12, 128, 128)` temporal state. Physical GPU1-3 all passed.

| Gate | Output max abs | State max abs | Finite/close |
| --- | ---: | ---: | --- |
| One step | 3.73e-9 | 2.98e-8 | yes |
| 1,000 repeated inputs | 2.61e-8 | 1.40e-6 | yes |
| 1,000 random q/k/v/decay/beta inputs | 1.49e-8 | 1.79e-7 | yes |

The random-sequence mean absolute differences were `2.71e-9` for output and
`1.78e-8` for state. All three GPUs reproduced the same bounds.

The production source also built into the installed vLLM package and passed
the random-sequence gate:

```text
size:   225816 bytes
sha256: 0a5c34b35e9508aaefd8dc5e1bd75436dcddc733b14ea5472f69e2db8a22c6f1
```

## Performance

The candidate's absolute median remained stable near 0.051 ms, but the
reference varied materially. Independent serial runs are the decision source:

| Run | Reference (ms) | Candidate (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Prototype | 0.078178 | 0.051833 | 1.5083x |
| Random-sequence repeat | 0.064786 | 0.050631 | 1.2796x |
| Production-source repeat | 0.064518 | 0.049102 | 1.3140x |

A simultaneous GPU1-3 run reported 3.22-3.79x because the reference medians
rose to 0.167-0.194 ms while candidate medians stayed at 0.051-0.052 ms. Those
concurrent ratios are excluded from the decision because host/framework
scheduling distorted only the multi-call reference path. They remain useful
as cross-device correctness and candidate-latency evidence.

Remote artifacts are untracked:

```text
/root/competition/bench_runs/20260715_E_GDN_07/result.json
/root/competition/bench_runs/20260715_E_GDN_07/random.json
/root/competition/bench_runs/20260715_E_GDN_07/production.json
/root/competition/bench_runs/20260715_E_GDN_07/gpu{1,2,3}.json
```

## Decision

`REJECT AS PERFORMANCE WINNER`. Two independent serial repeats achieve only
1.28x and 1.31x, below the development plan's 1.5x recurrent-update gate. Keep
the prototype and production integration commit on the experiment branch, but
do not merge `d8b5cd4` into `integration/perf-winners`, do not patch the remote
model, and do not spend a TP4/service qualification cycle on this version.

E-CAP-02 remains accepted: custom CoreX extensions are viable. A successor
must improve the kernel itself, such as vectorized memory access or BI tensor
instructions, before model integration is reconsidered.
