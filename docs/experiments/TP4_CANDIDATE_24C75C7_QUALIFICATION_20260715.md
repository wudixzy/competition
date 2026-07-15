# TP4 candidate 24c75c7 qualification

## Scope

This run qualified the complete candidate stack on a healthy four-card host
after a ModelHub evaluator allocation stalled during `init_device`:

```text
host:    ssh-a163074c.default.gpu.phanthy.com
runtime: 6e0e66c0427cacb1c65e3364d3d3c4bc23056a15
head:    24c75c7 (same runtime, startup diagnostics enabled)
model:   /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
TP:      4
context: 262144
```

Commit `24c75c7` changes only diagnostics and evidence relative to the tested
runtime: it enables existing startup-stage logging in the Docker image and
adds a group preflight. `computility-run.yaml` is unchanged.

## Build and startup

The Docker-equivalent `patch_ops.sh` completed successfully. All seven CoreX
extensions compiled, all package patches applied, and the final Python compile
gate passed.

Before model launch:

- four independent CUDA allocation/synchronize/256x256 matmul probes passed;
- four-rank NCCL all-reduce returned `10.0` on every rank;
- the vLLM-equivalent world/TP/PP NCCL+Gloo group sequence passed on every
  rank with no timeout.

The service reached HTTP health 200 and remained running as PID/PGID
`2949/2949`:

```text
GPU cache blocks: 16878
CPU cache blocks: 6553
maximum concurrency at 262144 tokens: 1.03x
```

## Correctness gates

| Gate | Result |
| --- | --- |
| Quick API smoke | 8/8 pass |
| Full API/multimodal/tool smoke | 15/15 pass |
| Forced decode | 1,000 tokens, finish=length |
| Forced-decode elapsed | 88.332 s |
| Forced-decode SHA256 | `1766c3c44bfb672e32b2e35419c5e06490e539e54250ab2fc1012c539e68835f` |
| Qualified-hash equality | exact |
| Service log | no Gloo/OOM/non-finite/worker loss/native crash |

Long-context cold/warm results:

| Prompt | Cold s | Warm s | Warm cached | Completion | Hash equality |
| ---: | ---: | ---: | ---: | ---: | --- |
| 99,500 | 154.536 | 18.247 | 99,296 | 8 | exact |
| 235,000 | 554.632 | 56.838 | 234,544 | 8 | exact |

Both prompt sizes produced the previously qualified message SHA256:

```text
a3dc73d02269b1b3682ed84197c3d2d0ddc39dfdb544f73fb3ea832f1fb30b4d
```

## Fixed single-concurrency sample

The local fixed sample used eight sequential requests, 1,809 prompt tokens and
64 completion tokens per request:

| Metric | Result | Competition gate |
| --- | ---: | ---: |
| Request success | 100% | >=99% |
| TTFT P90 | 1.795 s | <=5 s |
| Output TPS P10 | 15.5445 | >=20 |
| Output TPS total | 12.4442 | diagnostic |
| Input TPS | 351.7430 | dataset-dependent |
| Cache TPS | 307.6050 | dataset-dependent |
| Cache hit rate | 87.4516% | >=50% |
| Local weighted value | 1417.8730 | not official-comparable |

The local weighted value is not comparable with the official 8,000 target:
this short prompt sample does not reproduce the evaluator's long-input/cache
mix. Output TPS P10 is directly comparable in units and remains 4.4555 TPS
below the target. Reaching 20 from 15.5445 requires about 28.7% relative
throughput improvement.

## Artifacts

```text
/root/competition-candidate/build_a163.log
/root/competition-candidate/vllm_group_preflight_a163.json
/root/competition-candidate/service_a163.log
/root/competition-candidate/smoke_quick_a163.json
/root/competition-candidate/smoke_full_a163_v2.json
/root/competition-candidate/decode1000_a163.json
/root/competition-candidate/long_99500_a163/long_context_summary.json
/root/competition-candidate/long_235000_a163/long_context_summary.json
/root/competition-candidate/bench_fixed_24c75c7_a163.json
```

## Decision

`QUALIFIED FOR CORRECTNESS AND LONG CONTEXT; PERFORMANCE TARGET NOT MET`.
The complete candidate stack is stable and exact on a healthy TP4 host. Keep
it as the new integration baseline. Do not attribute the evaluator Gloo reset
to model code, and do not relax correctness to chase the remaining decode gap.
