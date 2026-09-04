# M1-176 FP16-QK focused full-model TP4 follow-up

## Decision

M1-162 FP16-QK passes the focused full-model TP4 development screen. One
control and one candidate service used the fixed full model, the same runtime,
launch arguments, request semantics and nine-request cold population. Candidate
dispatch was observed four times; control dispatch was zero.

Across nine paired TTFT samples, mean `control / candidate - 1` gain was
35.56%, with a one-sided 95% bootstrap lower bound of 30.66%. The 16K, 32K and
64K buckets were all stable. This clears the frozen 5% continuation rule and
authorizes 131K/235K cold confirmation followed by focused teacher-forced
distribution. It does not authorize a default selector, `main`, YAML,
near-262K, capability, cache, full protocol or formal 881 run.

Safe structured evidence is
`docs/experiments/evidence/M1_176_FP16_QK_FOCUSED_TP4_20260904/summary.json`.

## Reviewer gate repairs

- Generic v2 validity now rejects empty identities, non-finite timing and
  inconsistent request counts.
- Capability is checked only when called; every configured stratum must then
  contain samples, internally consistent paired outcomes, a finite lower CI,
  bootstrap and exact McNemar diagnostics.
- A negative finite performance gain is candidate `fail`, not an exception
  converted into `invalid`.
- The historical L3 qualifier now binds distribution source/runtime/model/
  instance, targets, sample population and arm roles, then recomputes the
  decision. Unknown state, empty A/A, unbound candidate or decision mismatch is
  `invalid`.
- TTFT ends at first output. TPOT and Output TPS over first-to-last output use
  `completion_tokens - 1` intervals.
- The focused `attention_operator` runner has only control and candidate; it
  does not run control B, cache branches, capability, HMAC identity, tree hash
  or per-file SHA-256.

Gate commits were `9bbe3bb`, `a280966`, `61f5cd7`, `4e2b2e7` and `4d79d00`.
Both measured arms ran exact source
`4d79d00fce24aa1ba6a515870653341f608242ff` with a clean tree. Later harness
commit `e75470b` changes only the active-listener port probe and was not used by
either measured arm.

## Retained operator evidence

Capture and head mapping were not rerun. Existing M1-176 evidence remains an
operator development screen:

- frozen L1 synthetic 16K/32K/64K G2: pass;
- TP1-derived real activation replay, 12 rank-local cells: G2 pass;
- mapping: Q0-3/KV0, Q4-7/KV0, Q8-11/KV1, Q12-15/KV1;
- L1 kernel medians, baseline/candidate: 72.103/62.606 ms at 16K,
  136.101/116.755 ms at 32K and 264.147/225.297 ms at 64K;
- L2 maximum relative-L2 and max-absolute error ratios: 1.0000123 and
  1.0017094; all values finite and repeats exact.

These do not claim real TP4 model numerics. This follow-up adds real full-model
TP4 service timing and dispatch evidence.

## Runtime and workload

- instance: `cc-ce19242b-436f-4141-868b-610eb3ac8cee-0`;
- four Iluvatar BI-V100, each 34,057,748,480 bytes free at session preflight;
- Python 3.10.12, CoreX 3.2.3, Torch 2.1.0, vLLM 0.6.3,
  Transformers 4.55.3, clang 16.0.6;
- runtime: `overlay-44a43f355cfc-torch-2.1.0-vllm-0.6.3`;
- model/tokenizer: `/root/public-storage/models/Qwen/Qwen3.6-35B-A3B`;
- TP4, FP16, `max_model_len=262144`, block size 16,
  `max_num_batched_tokens=8192`, `max_num_seqs=1`;
- 16K/32K/64K cold prompts, three repetitions, eight-token fixed greedy
  completion, temperature zero and seed 20260904.

The runtime-compatible module was rebuilt from the frozen M1-162 CUDA source
under production module name `corex_fused_paged_prefill`. Both arms used the
same temporary overlay; selector state was their sole difference. No SHA-256
was required or recorded for this same-host build.

One reusable preflight (`m1-176-focus-preflight-4e2b2e7`) passed a 1024-square
FP16 matmul on all cards and TP4 NCCL all-reduce value 10.0 on ranks 0-3; all
children were reaped.

## TP4 service timing

All 18 requests returned HTTP 200 with complete SSE, usage,
`finish_reason=length`, exact prompt count and eight completion tokens.

| Prompt | Control raw TTFT s | Candidate raw TTFT s | Mean paired gain |
| ---: | --- | --- | ---: |
| 16K | 29.479, 26.374, 28.238 | 22.540, 19.593, 19.468 | 36.81% |
| 32K | 56.245, 56.064, 56.171 | 38.840, 38.920, 38.599 | 44.80% |
| 64K | 102.649, 101.139, 101.558 | 81.350, 81.511, 81.304 | 25.06% |

Aggregate paired gain was 35.56%; one-sided 95% lower CI was 30.66% from
20,000 bootstrap resamples. Every bucket had positive mean and no individual
regression over 5%, so qualification is `pass`. No control B or reversed pair
was collected because the result was outside the 2%-5% gray zone.

The eight-token decode portion is an immediate-error/reporting diagnostic, not
a long-decode throughput claim. Output TPS is
`7 / (last_token_time - first_token_time)`.

## Lifecycle, limitations and next stage

Both measured services used scoped process groups. Cleanup sent TERM, waited
under the 60-second grace period, did not send KILL, and left zero live service
processes. Postflight found no API server, worker or GPU process. Fatal scans
found zero OOM, CUDA error, segfault, timeout, collective/state error or worker
loss.

Two setup-only invalid attempts are excluded: one compiler probe failed before
service startup; one candidate attempt met socket TIME_WAIT before startup.
Neither loaded a model or sent a request.

Capability strata, partial-prefix/cache transparency, full API protocol,
teacher-forced distribution, 131K/235K, near-262K and formal 881 were not run.
The only next authorized work is fixed 131K/235K cold confirmation on this
candidate, then focused A/A-calibrated teacher-forced distribution. Promotion
still requires reviewer approval and cumulative final integration gates.
