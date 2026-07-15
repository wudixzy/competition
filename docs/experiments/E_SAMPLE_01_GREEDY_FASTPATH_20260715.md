# E-SAMPLE-01 greedy sampler probability-transform bypass

## Scope

E-SAMPLE-01 tested an exact fast path for pure greedy requests that do not
request output or prompt log probabilities. The candidate applied the existing
minimum-token and repetition/frequency/presence penalties, then sampled by
directly taking the argmax of the resulting FP16 logits. It skipped the
otherwise unused FP32 conversion, temperature division, top-k/top-p/min-p
filtering, softmax, and log-softmax operations.

The path was disabled for random or beam sampling, requested logprobs,
speculative/deferred sampler output, and GPU probability tensors. The runtime
switch was `BI100_SAMPLER_GREEDY_FASTPATH`.

## Four-GPU microbenchmark

Each GPU ran 1,000 random comparisons and three explicit tie cases. Candidate
token IDs were exact in every case. Nine trials of 1,000 iterations over a
151,936-wide FP16 vocabulary produced:

| GPU | Reference ms | Candidate ms | Speedup | Saving ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.327085 | 0.047726 | 6.853x | 0.279359 |
| 1 | 0.327638 | 0.047760 | 6.860x | 0.279878 |
| 2 | 0.327560 | 0.047830 | 6.848x | 0.279730 |
| 3 | 0.327305 | 0.047687 | 6.864x | 0.279618 |

The isolated saving projected to only about 0.44% of the approximately
63 ms/token production decode path, so a strict service A/B was required.

## Correctness gates

The candidate used the production TP4 command with 262,144 context, chunked
prefill, prefix caching, and `ENABLE_CUSTOM_IPC=1`.

| Gate | Result |
| --- | --- |
| Service health | HTTP 200 |
| Existing full API/multimodal/tool smoke | 15/15 pass |
| Random, output-logprobs, prompt-logprobs fallbacks | 3/3 pass |
| Forced decode | 1,000 tokens, finish=length |
| Forced-decode elapsed | 65.286 s |
| Forced-decode SHA256 | `1766c3c44bfb672e32b2e35419c5e06490e539e54250ab2fc1012c539e68835f` |
| Qualified-hash equality | exact |
| Service log | no ERROR/Traceback/Gloo/NCCL/native crash |

The 262K startup profile is unusually slow but not hung. A known-good service
spent 6 minutes 29 seconds in `determine_num_available_blocks`; the candidate
became healthy after 7 minutes 59 seconds total. Startup monitoring must allow
at least 12 minutes before classifying this stage as stalled.

## Strict paired TP4 result

Candidate and baseline used the same patched binary, service arguments, prompt
salts, seed, request count, and cache setup. The baseline changed only
`BI100_SAMPLER_GREEDY_FASTPATH=0`.

| Pair | Candidate Output TPS P10 | Baseline Output TPS P10 | Change |
| ---: | ---: | ---: | ---: |
| 1 | 15.8801 | 15.9539 | -0.46% |
| 2 | 15.9404 | 15.7994 | +0.89% |
| 3 | 15.9489 | 16.0044 | -0.35% |
| Mean | 15.9231 | 15.9192 | +0.02% |

All six runs had 100% request success and the service remained healthy. Two of
three pairs regressed and the mean difference was within normal run noise. The
isolated softmax saving therefore did not produce a repeatable end-to-end gain.

## Artifacts

```text
/root/e_sample_01/results/gpu0.json
/root/e_sample_01/results/gpu1.json
/root/e_sample_01/results/gpu2.json
/root/e_sample_01/results/gpu3.json
/root/e_sample_01/fallback_api.json
/root/e_sample_01/smoke_full.json
/root/e_sample_01/sustained_1000.json
/root/e_sample_01/bench_candidate_1.json
/root/e_sample_01/bench_candidate_2.json
/root/e_sample_01/bench_candidate_3.json
/root/e_sample_01/bench_baseline_1.json
/root/e_sample_01/bench_baseline_2.json
/root/e_sample_01/bench_baseline_3.json
/root/e_sample_01/service_candidate_retry.log
/root/e_sample_01/service_baseline_final.log
```

## Decision

`REJECTED FOR MAIN`: correctness passed and the isolated operator was much
faster, but the production TP4 comparison failed the requirement that all
three paired Output TPS P10 results improve. The implementation remains on
`exp/E-SAMPLE-01-greedy-fastpath` as evidence; production `main` and the remote
runtime were restored to the original sampler.
