# E-MOE-20 direct routed-expert prototype

## Scope

E-MOE-20 replaces the current T=1 selected-weight gather, W13 GEMM, W2 BMM,
and routed reduction with shape-specific direct-addressing kernels. It does
not change routing and does not alter prefill. This first result is a
single-GPU algorithm gate, not a production or TP4 result.

Tested rank-local shape:

```text
experts=256, top_k=8, hidden=2048, intermediate=128, dtype=FP16
```

The staged candidate uses direct W13, the existing `SiluAndMul`, then direct
W2 plus routed-weight reduction. The fused candidate also folds activation
into the W13 stage.

## GPU1 result

Physical GPU1 on `ssh-9cd6a034` passed compilation, short smoke, nine timing
repeats, and a 500-step random numerical sequence. Only that physical GPU was
visible to the process.

| Boundary | Baseline ms | Staged ms | Staged speedup | Fused ms | Fused speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed expert path | 0.263837 | 0.043937 | 6.005x | 0.037477 | 7.040x |
| Routing plus expert path | 0.331490 | 0.112576 | 2.945x | 0.102432 | 3.236x |

Component medians explain the gain:

| Current component | ms | Direct replacement | ms |
| --- | ---: | --- | ---: |
| selected W13/W2 gather | 0.070440 | removed | 0 |
| W13 | 0.132896 | direct W13 | 0.020410 |
| activation | 0.010661 | retained by staged | 0.010661 |
| W2 | 0.053773 | direct W2 plus reduce | 0.016491 |
| exact reduce | 0.007478 | included above | 0 |

Both candidates exceeded the predefined `1.5x` fixed and `1.25x` routed
gates. No launch-parameter scan was needed.

## Numerics

All fixed and sequence outputs were finite, but neither candidate was
bit-exact because the custom GEMV reduction order differs from vendor GEMM.

| Candidate | Fixed max abs | 500-step finite | Sequence max abs | Mean abs |
| --- | ---: | ---: | ---: | ---: |
| Staged | 0.00012207 | 500/500 | 0.00012207 | 6.7841e-06 |
| Fused activation | 0.00012207 | 500/500 | 0.00024414 | 1.5606e-05 |

The fused activation intermediate reached `0.001953125` max absolute error,
while direct W13 reached `0.00024414`. The staged candidate is therefore the
only production candidate despite being about 10% slower on the routed
microbenchmark. It retains the vendor activation and has lower endpoint drift.

## Projection

Staged saves `0.218914 ms/layer` in this fixed-shape benchmark. If all 40 MoE
layers realize that saving, the static projection is about `8.76 ms/token`.
Applied mechanically to the qualified `15.8206 TPS` baseline, that corresponds
to roughly `18.4 TPS`. This is an upper-bound planning estimate, not a measured
service result, and it still falls short of 20 TPS by about 9%.

## Decision

`PASS SINGLE-GPU ALGORITHM GATE` for staged only. Next gates are:

1. reproduce timing and error bounds on physical GPUs 2 and 3;
2. add a default-off, strictly guarded production extension and unit tests;
3. wait for a four-GPU-qualified host before token-hash, multimodal, long
   decode, 262K startup, and strict service A/B.

Do not integrate the activation-fused variant and do not tune launch geometry.

## Cross-device reproduction

Physical GPUs 2 and 3 reproduced the result with the same 500-step numerical
statistics as GPU1:

| GPU | Staged fixed | Staged routed | Fused fixed | Fused routed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6.005x | 2.945x | 7.040x | 3.236x |
| 2 | 5.928x | 2.924x | 6.987x | 3.198x |
| 3 | 5.623x | 2.892x | 6.543x | 3.181x |

All three staged runs reported 500/500 finite steps, `0.00012207` sequence
max absolute error, and `6.7841e-06` mean absolute error. This satisfies the
cross-device gate and authorizes a default-off production dispatch on the
experiment branch.

## Production dispatch probe

Commit `c228378` adds a fixed-shape production extension and a default-off
`BI100_MOE_COREX_DIRECT_ROUTED` dispatch. The probe extracts the real
`_pure_pytorch_experts` method, uses the current E-MOE-13 plus exact-reduce
path as baseline, and changes only that global dispatch flag.

Physical GPU1 result:

```text
baseline median  0.325494 ms
direct median    0.106735 ms
speedup          3.0495x
fixed max abs    0.00006104
sequence finite  500/500
sequence max abs 0.00012207
sequence mean    0.0000067844
```

Static tests passed and all three production extensions compiled. This closes
the single-GPU production-dispatch gate. The feature remains default-off and
must not enter `main` before TP4 endpoint qualification.

Remote artifacts:

```text
/root/E_MOE_20/result.json
/root/E_MOE_20/bench.log
/root/E_MOE_20/smoke.json
/root/E_MOE_20/smoke.log
/root/E_MOE_20/logs/build_*.log
/root/E_MOE_20/result_gpu2.json
/root/E_MOE_20/result_gpu3.json
/root/E_MOE_20/bench_gpu2.log
/root/E_MOE_20/bench_gpu3.log
/root/competition-candidate/production/result_gpu1.json
/root/competition-candidate/production/bench_gpu1.log
/root/competition-candidate/production/smoke_gpu1.json
/root/competition-candidate/production/smoke_gpu1.log
```

## TP4 endpoint qualification

The old healthy four-card host ran a strict same-source A/B. Both services
used the same model source, protocol fixes, extension binary, evaluator
command, and cache settings. The only service change was:

```text
BI100_MOE_COREX_DIRECT_ROUTED=0 -> 1
```

Three paired fixed workloads produced:

| Pair | flag=0 Output TPS P10 | flag=1 Output TPS P10 | Relative gain |
| ---: | ---: | ---: | ---: |
| 1 | 15.1293 | 20.0563 | +32.57% |
| 2 | 15.0863 | 19.0185 | +26.07% |
| 3 | 15.1491 | 20.0312 | +32.23% |

All six runs completed 8/8 requests. Candidate TTFT P90 was
`2.134/2.214/2.173 s`, cache hit rate was `86.87%`, and the candidate service
log added no traceback, segmentation fault, OOM, or fatal event. A later N=2
diagnostic reached `19.9119 TPS`, but had a 99.28% cache hit rate and is not
part of the strict paired result.

The candidate also passed:

| Gate | Result |
| --- | --- |
| Full API and multimodal smoke | 15/15 |
| Dataset-shaped Agent matrix | 9/9 |
| Sustained greedy decode | 1,000 tokens, 51.930 s, finite |
| Sustained output hash | `1766c3c44bfb672e32b2e35419c5e06490e539e54250ab2fc1012c539e68835f` |
| 99.5K cold/warm | 157.252/16.583 s, 99,296 cached, exact output |
| 235K cold/warm | 562.368/55.489 s, 234,544 cached, exact output |
| Long-context output hash | `a3dc73d02269b1b3682ed84197c3d2d0ddc39dfdb544f73fb3ea832f1fb30b4d` |

The sustained hash is identical to flag=0 and the previously qualified
baseline. The long-context hashes are identical between cold and warm runs.

## Final decision

`ACCEPT AS TP4 PERFORMANCE WINNER`.

The service gain is positive and above 25% in all three strict pairs, while
functional, multimodal, Agent, sustained-decode, and 235K gates pass. Enable
the knob in `computility-run.yaml`, retain the code-level default-off fallback,
and merge the qualified implementation to `main`.

This does not prove the official Output TPS P10 target is stable: two local
pairs are just above 20 and one is 19.02. The next optimization must improve
the remaining margin rather than tune evaluator parameters. E-GDN-14 packed
decode is the next algorithmic candidate because it covers a measured
`0.165 ms/layer` boundary and has an independent stop rule.
