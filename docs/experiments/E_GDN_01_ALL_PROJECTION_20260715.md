# E-GDN-01: Fuse all Gated DeltaNet input projections

Date: 2026-07-15

## Hypothesis

Each of the 30 Gated DeltaNet layers executes four input projections for every
prefill and decode token. Replacing the separate QKV, gate, beta, and decay
linears with one tensor-parallel merged linear should reduce launch and GEMM
overhead without changing the fixed evaluator contract or model output.

## Manifest

```text
baseline commit: bd303c09bc17d7e568460b2a87fddcd39ca6c088
candidate commit: f623edbd4206518a031c550625c2378fb9ecf8e2
branch: exp/E-GDN-01-all-projection
model: /root/public-storage/models/Qwen/Qwen3.6-35B-A3B
hardware: 4 x BI-V100-50C-200G, TP=4
seed: 123
```

## Change

`qwen3_6_scripts/qwen3_5.py` replaces `in_proj_qkv`, `in_proj_z`,
`in_proj_b`, and `in_proj_a` with one `MergedColumnParallelLinear`. The output
shards remain independently tensor-parallel sharded in this order:

```text
[q, k, v, z, beta, decay]
```

The checkpoint loader maps the original fused QKV tensor to shards 0-2 and the
three remaining projection tensors to shards 3-5. The forward path performs
one linear and splits its output into the original views. Rollback is a branch
switch to `main`; no checkpoint or evaluator configuration changes are needed.

## Contract

- `computility-run.yaml` changed: no
- fixed evaluator arguments changed: no
- performance environment overrides: none
- debug or profiler enabled for score: no

## Correctness gates

- loader unit tests: 4/4 pass
- complete non-GPU unit discovery: 122 pass, 1 skipped
- four-GPU CUDA preflight: pass
- four-GPU NCCL preflight: pass
- deterministic API oracle: 3/3 exact, including content, reasoning,
  tool calls, finish reason, and usage
- quick smoke: 7/7 pass
- full smoke: 15/15 pass in 100.06 seconds
- 99,500-token cold/warm boundary: pass; exact output equality
- candidate service error-log scan: zero `ERROR`
- candidate API health after tests: HTTP 200

## Primitive results

The probe used the real tensor-parallel rank shapes on all four devices. All
outputs were finite and exactly equal to the four-linear reference.

| Shape | Four linears (ms) | One merged linear (ms) | Speedup |
| --- | ---: | ---: | ---: |
| decode, T=1 | 1.6526-1.6760 | 0.6103-0.6242 | 2.68-2.71x |
| prefill sample, T=64 | 0.6692-0.6781 | 0.2871-0.2956 | 2.27-2.36x |

Artifacts:

```text
bench_runs/20260715_E_GDN_01/projection/gpu0.json
bench_runs/20260715_E_GDN_01/projection/gpu1.json
bench_runs/20260715_E_GDN_01/projection/gpu2.json
bench_runs/20260715_E_GDN_01/projection/gpu3.json
```

## Service qualification

The candidate reached HTTP readiness in 7 minutes 34 seconds. Weight loading
completed normally, CUDA/NCCL preflights passed, and the runtime reported
16,871 GPU blocks versus 16,884 for the stable baseline (13 fewer, about
0.08%). The candidate remained healthy through oracle, smoke, and performance
tests.

## Strict performance pair 1

Both runs used eight serial streaming requests, 64 generated tokens, seed 123,
prompt repeat 126, and the exact salt
`bd303c0-baseline-A-20260715`. Token totals matched exactly: 14,520 prompt,
1,864 uncached prompt, 12,656 cached, and 512 completion tokens.

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Success rate | 100% | 100% | equal |
| TTFT P90 (s) | 1.8080 | 1.9545 | +8.10% |
| Decode TPS P10 | 12.6935 | 13.0256 | +2.62% |
| E2E output rate P10 | 9.7036 | 9.7676 | +0.66% |
| Input TPS | 294.2875 | 292.1358 | -0.73% |
| Cache TPS | 256.5084 | 254.6330 | -0.73% |
| Cache hit rate | 87.1625% | 87.1625% | equal |
| Weighted overlap score | 1180.5548 | 1179.0611 | -0.13% |
| Disjoint score | 462.5878 | 466.3434 | +0.81% |

The first attempted comparison used a shorter salt and therefore had nine
fewer prompt tokens per request. It is retained only as diagnostic evidence
and is excluded from the strict decision.

## Additional paired comparisons

Stable and candidate services were each cold-started once after verifying the
Git branch, commit, and installed runtime file SHA256. They then ran salts
`E-GDN-01-PAIR-2-20260715` and `E-GDN-01-PAIR-3-20260715` in the same order.
Each side reported 14,520 prompt, 1,864 uncached prompt, 12,656 cached, and 512
completion tokens in both pairs.

| Metric | P2 baseline | P2 candidate | Change | P3 baseline | P3 candidate | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TTFT P90 (s) | 2.7348 | 2.6706 | -2.35% | 2.0430 | 1.9489 | -4.61% |
| Decode TPS P10 | 11.4082 | 12.9718 | +13.71% | 11.4180 | 13.0651 | +14.43% |
| E2E output rate P10 | 8.2898 | 9.2997 | +12.18% | 8.6118 | 9.7963 | +13.75% |
| Input TPS | 258.1762 | 278.7393 | +7.96% | 266.0155 | 292.6016 | +9.99% |
| Cache TPS | 225.0330 | 242.9563 | +7.96% | 231.8659 | 255.0389 | +9.99% |
| Weighted overlap score | 1040.2664 | 1134.1218 | +9.02% | 1066.1999 | 1181.2553 | +10.79% |
| Disjoint score | 410.3991 | 454.0873 | +10.65% | 417.2073 | 467.4013 | +12.03% |

The median change across all three strict pairs is +13.71% for Decode TPS P10
and +9.02% for weighted overlap score. Decode TPS improves in all three pairs;
the short-output weighted score is neutral in pair 1 and improves in pairs 2
and 3. Pair 1 used already-running services while pairs 2 and 3 used fresh
stable and candidate startups, so the spread also records material
cross-startup device variability.

An automation attempt produced `baseline-P2.json` and `baseline-P3.json` while
the installed runtime SHA256 still matched the candidate. Those files are
invalid, are excluded from every table and decision, and are superseded by the
`*-valid.json` artifacts.

Valid artifacts:

```text
bench_runs/20260715_PERF/baseline-A.json
bench_runs/20260715_E_GDN_01/candidate-B2.json
bench_runs/20260715_E_GDN_01/baseline-P2-valid.json
bench_runs/20260715_E_GDN_01/candidate-P2-valid.json
bench_runs/20260715_E_GDN_01/baseline-P3-valid.json
bench_runs/20260715_E_GDN_01/candidate-P3-valid.json
```

## Long-context boundary

The final gate ran immediately after a fresh candidate restart so the first
request could not inherit any KV cache. The script constructed exactly 99,500
prompt tokens and requested up to 16 output tokens under the fixed 100,000-token
model limit.

| Request | Prompt tokens | Cached tokens | Completion tokens | Elapsed (s) |
| --- | ---: | ---: | ---: | ---: |
| cold | 99,500 | 0 | 8 | 159.464 |
| warm | 99,500 | 99,296 | 8 | 17.539 |

Both requests ended with `stop` and produced the same message SHA256:

```text
a3dc73d02269b1b3682ed84197c3d2d0ddc39dfdb544f73fb3ea832f1fb30b4d
```

The raw messages, finish reasons, completion counts, and hashes are equal. The
warm request exceeds the required 98,304 cached tokens. A standalone verifier
replayed every assertion from the result files and exited 0; the candidate
service remained HTTP 200 with zero fatal log matches.

Artifacts:

```text
bench_runs/20260715_E_GDN_01/long-99500-cold/long_context_response1.json
bench_runs/20260715_E_GDN_01/long-99500-cold/long_context_response2.json
bench_runs/20260715_E_GDN_01/long-99500-cold/long_context_summary.json
bench_runs/20260715_E_GDN_01/server-candidate-long-cold.log
```

## Decision

`KEEP AS PERFORMANCE WINNER`. Three strict token-matched pairs show a
repeatable decode improvement, with a +13.71% median Decode TPS P10 change and
+9.02% median weighted-score change. Full smoke, deterministic oracle,
preflights, the 99,500-token cold/warm boundary, and log scans all pass. Keep
this commit on `exp/E-GDN-01-all-projection` as the qualified winner for the
next integration cycle; stable `main` remains unchanged until winner
integration regression testing.
