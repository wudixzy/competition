# M1-180 three-arm capability adjudication and baseline decomposition

## M1-181 reviewer addendum

The historical M1-180 evidence JSON is retained byte-for-byte, but four gate
interpretations in the original report are superseded:

- The 60-case population establishes only that the basic deterministic
  functional contracts exercised by that harness did not regress. It does not
  establish statistical task capability. In particular, the reasoning prompt
  disclosed its expected `FINAL` answer, most strata were simple exact
  contracts, and long-context coverage stopped at 16K.
- Fused-off versus M1-109 and fused-off versus M1-162 have no independently
  started fused-off A/A control. They remain useful uncalibrated distribution
  diagnostics, but the M1-109 A/A envelope cannot calibrate a comparison whose
  left control is fused-off.
- M1-109 versus M1-162 remains a valid A/A-calibrated incremental distribution
  drift result because its left control is M1-109 and the reused M1-179 A/A is
  identity-matched to M1-109.
- The six-pair M1-109-to-M1-162 TTFT mean of +2.2763%, with no run-level CI and
  one negative 16K pair, is
  `positive_diagnostic_underpowered / inconclusive`. The historical machine
  field `incremental_performance.status=pass` must not be interpreted as a
  performance gate pass.

Future harness output separates zero-regression deterministic contracts from
statistical capability, validates the reasoning/content protocol without
putting the answer in the prompt, uses whole-term multimodal color matching,
and binds every distribution calibration envelope to the comparison's left
control variant.

## Decision

The valid three-arm result is
`development_capability_screen_passed_but_strata_underpowered / inconclusive`.
All 60 frozen development cases passed on fused-off, M1-109 and M1-162, so
this screen found no deterministic capability regression in either fused
increment. Each required stratum contains only ten pairs, however, and is
underpowered for promotion. This is not a capability-gate pass.

Both attention increments also show real distribution differences relative to
the reused exact-zero M1-109 A/A envelope. Fused-off versus M1-109 has nine
top-1 flips, five high-margin flips and local positive NLL deltas at 4K, 16K
and 32K. M1-109 versus M1-162 has eleven top-1 flips, eight high-margin flips
and local positive NLL deltas at 4K and 16K. The decomposition therefore shows
that M1-178's combined-path difference was not created solely by FP16-QK:
M1-109 already differs materially from fused-off, and FP16-QK adds a second
material distribution change.

M1-162 remains a high-performance development candidate, but production
promotion stays blocked pending reviewer-directed capability expansion. M1-109
also remains only a development candidate pending its own full gates. No
default selector, YAML, `main`, formal 881 workload or new kernel was changed.

The repository-safe machine summary is
`docs/experiments/evidence/M1_180_THREE_ARM_CAPABILITY_ADJUDICATION_20260905/summary.json`.
Private observations, prompts, synthetic images, token identities and service
logs remain only under the remote `/tmp` experiment root.

## Implementation and identity

The M1-180 harness adds:

- one adaptive capability/distribution workload in
  `tests/m1_180_capability_distribution_api.py`;
- a three-arm orchestrator in
  `scripts/run_m1_180_three_arm_adjudication.py`;
- strict arm/runtime/variant binding and repository-safe aggregation in
  `tests/compare_m1_180_adjudication.py`;
- additional privacy-safe flip direction diagnostics in the M1-179
  comparator, without rewriting M1-179 evidence.

The valid run used:

- branch: `exp/M1-132-layered-quality-gate-20260729`;
- evidence source: `477d18133cc93d370e58d04b3994f87990b9e614`;
- source dirty summary: `clean`;
- instance: `cc-ce19242b-436f-4141-868b-610eb3ac8cee-0`;
- GPU: four Iluvatar BI-V100 devices, each with 34,057,748,480 bytes free at
  the reused session preflight;
- model/tokenizer:
  `/root/public-storage/models/Qwen/Qwen3.6-35B-A3B`;
- runtime overlay:
  `/tmp/m1-176-focused-runtime.ihZwyu/runtime/site-packages`;
- runtime: Python 3.10.12, CoreX 3.2.3, Torch 2.1.0, vLLM 0.6.3,
  Transformers 4.55.3 and clang 16.0.6;
- reused preflight: `m1-176-focus-preflight-4e2b2e7`, with four single-GPU
  FP16 matmuls and the TP4 collective qualified;
- M1-109 extension:
  `/tmp/m1-179-build-653b90b/corex_fused_paged_prefill_m1_109_fp32_qk.so`,
  SHA-256
  `b7b30f8c3c3af0153c58dde4760159dbdfeeec17fd352b192ae57990ad1a0be8`;
- M1-162 extension:
  `/tmp/m1-179-build-653b90b/corex_fused_paged_prefill_m1_162_fp16_qk.so`,
  SHA-256
  `3724d6651eed814b84043d0b0155cfb381baab02e9db56665116dc8e016f2f91`.

Only the two precompiled extension binaries retain SHA-256 identity. No source,
report or overlay-tree hash gate was added.

The arms ran sequentially with TP4, FP16, `max_model_len=262144`, block size
16, `max_num_batched_tokens=8192` and the same server command and unrelated
environment. Their bound identities were:

| Arm | Fused selector | Runtime variant | Dispatch count |
| --- | ---: | --- | ---: |
| fused-off | 0 | `fused_off` | 0 |
| M1-109 | 1 | `m1_109_fp32_qk` | 4 |
| M1-162 | 1 | `m1_162_fp16_qk` | 4 |

Runtime introspection confirmed the loaded extension path for both fused arms;
the conclusion does not rely on selector environment variables alone.

## Frozen capability screen

The self-contained frozen set is defined in the repository harness and does
not download data or use candidate generations as answer keys. Each stratum
contains ten independently validated cases:

- code: exact stdout for short Python snippets;
- reasoning: fixed arithmetic and logic answers;
- tools: forced function name plus exact JSON arguments;
- structured output: exact values under a JSON schema;
- multimodal: generated solid-color PNGs with independently known labels;
- long context: exact marker recall at 4K, 8K and 16K token lengths.

Every arm completed 60 capability requests. The first four cases in each
stratum formed the 24-pair smoke. M1-162 had zero baseline-only smoke failures
against both fused-off and M1-109, so the same service continued to ten cases
per stratum. HTTP/response/finish contracts and all finite checks passed;
finish reasons were the expected `stop`, `length` or `tool_calls`, and all
capability requests reported `cached_tokens=0`.

All three paired comparisons have the same result:

| Comparison | Both pass | Baseline only | Candidate only | Both fail | Paired difference | One-sided bootstrap lower | McNemar exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fused-off vs M1-109 | 60 | 0 | 0 | 0 | 0 pp | 0 pp | 1.0 |
| M1-109 vs M1-162 | 60 | 0 | 0 | 0 | 0 pp | 0 pp | 1.0 |
| fused-off vs M1-162 | 60 | 0 | 0 | 0 | 0 pp | 0 pp | 1.0 |

The aggregate development lower bound exceeds the predeclared -5 percentage
point margin. Nevertheless, every individual stratum is 10/10 and explicitly
underpowered; aggregate success cannot hide an unmeasured stratum regression.
The capability layer is therefore `inconclusive`, not `pass`.

## Distribution decomposition

Each arm appended one cold teacher-forced request at 4K, 16K, 32K and 64K,
with 64 fixed positions per length. All 12 requests were HTTP 200, used the
same shared in-memory identity key, had finite logprobs and reported
`cached_tokens=0`. The key and raw identities were neither printed nor
persisted in the repository.

The M1-179 M1-109 control-A/control-B envelope was reused because instance,
model, runtime overlay, TP/dtype, request targets, position count, service
configuration and M1-109 extension binary identity matched. That independently
started A/A population was exact at all 256 positions: no flips and zero
teacher/shared logprob or NLL difference.

| Comparison | Top-1 agreement | Flips / high-margin | Mutual top-k | Teacher/shared P99 (nats) | Mean NLL delta | Position upper diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fused-off vs M1-109 | 96.484375% | 9 / 5 | 0.777778 | 7.684062 / 5.505184 | +0.083451 | +0.226930 |
| M1-109 vs M1-162 | 95.703125% | 11 / 8 | 0.909091 | 5.776591 / 4.591725 | -0.083481 | +0.043299 |
| fused-off vs M1-162 | 96.875000% | 8 / 5 | 0.875000 | 5.947510 / 4.349002 | -0.000031 | +0.128668 |

The high-margin threshold is 0.1 nat and the NLL diagnostic threshold is 0.01
nat because the matched A/A envelope is zero. All three comparisons are
`distribution_drift_requires_adjudication / inconclusive`; top-1 agreement is
reported diagnostically and no uniform 98% threshold is used.

### Fused-off to M1-109

| Length | Candidate/control/equal positions | Mean NLL delta | Median NLL delta |
| ---: | ---: | ---: | ---: |
| 4K | 36 / 27 / 1 | +0.137226 | -0.010324 |
| 16K | 26 / 37 / 1 | +0.158067 | +0.020158 |
| 32K | 24 / 39 / 1 | +0.174768 | +0.054922 |
| 64K | 36 / 27 / 1 | -0.136258 | -0.030397 |

The first divergent sampled position was length 4K, position 145, with a
0.046875-nat fused-off margin. Four of the nine flips had the teacher token as
the fused-off top-1 and none promoted it to M1-109 top-1. These directional
facts describe logits movement; they do not establish capability loss.

### M1-109 to M1-162

| Length | Candidate/control/equal positions | Mean NLL delta | Median NLL delta |
| ---: | ---: | ---: | ---: |
| 4K | 29 / 34 / 1 | +0.052221 | +0.001406 |
| 16K | 35 / 28 / 1 | +0.096181 | -0.011498 |
| 32K | 31 / 32 / 1 | -0.164066 | approximately 0 |
| 64K | 34 / 29 / 1 | -0.318260 | -0.015491 |

The first divergent sampled position was length 4K, position 178, with a
0.460938-nat M1-109 margin. None of the eleven flips demoted a teacher token
that was M1-109 top-1; five promoted the teacher token to M1-162 top-1. The
safe evidence records both margins, cross-top-k ranks and teacher-logprob
direction for every flip, without token IDs or text. Again, this is not a
capability-gain claim.

Position bootstrap values only describe uncertainty from the fixed sampled
positions inside the four fixed length strata. They are not service-level or
run-to-run confidence intervals, and positive and negative length effects are
not allowed to cancel the per-length adjudication.

## Incremental TP4 timing diagnostic

Timing ran only because the M1-162 smoke had no baseline-only failure. M1-109
and M1-162 each issued two cold requests at 16K, 32K and 64K; all six pairs had
`cached_tokens=0`.

| Length | Repetition | M1-109 TTFT (s) | M1-162 TTFT (s) | Gain |
| ---: | ---: | ---: | ---: | ---: |
| 16K | 0 | 20.101664 | 20.015239 | +0.4318% |
| 16K | 1 | 19.563457 | 19.595850 | -0.1653% |
| 32K | 0 | 40.416385 | 38.716847 | +4.3897% |
| 32K | 1 | 39.810942 | 39.539035 | +0.6877% |
| 64K | 0 | 85.036945 | 81.475557 | +4.3711% |
| 64K | 1 | 84.818785 | 81.601548 | +3.9426% |

The mean of the six paired gains is +2.2763%. This is positive but small and
contains one negative 16K sample; with two samples per length and no run-level
confidence analysis it is a diagnostic, not a performance promotion pass. It
does preserve M1-162 as a potentially valuable candidate while capability and
distribution adjudication remain open.

## Request budget, invalid attempts and lifecycle

The valid r3 population used exactly three service startups and 204 model
requests:

- fused-off: 60 capability + 4 teacher-forced = 64;
- M1-109: 60 capability + 4 teacher-forced + 6 timing = 70;
- M1-162: 60 capability + 4 teacher-forced + 6 timing = 70.

Valid-run wall time was 4,631.000 seconds. Capability/distribution/timing work
inside each service took 792.691, 1,081.321 and 1,064.404 seconds respectively.

Two earlier harness attempts are invalid and are not evidence:

1. `/tmp/m1-180-evidence-7883a8c` started fused-off once and completed 60
   capability requests, then the distribution probe correctly rejected the
   new combined workload manifest. Commit `477d181` fixed that compatibility
   bug; the arm cleaned up and no later arm started.
2. `/tmp/m1-180-evidence-477d181` completed fused-off and M1-109, but the
   non-detached orchestrator lost its output channel at an arm boundary. The
   in-memory identity key was consequently lost before M1-162, so the whole
   round is invalid and its observations were not reused. Explicit postflight
   was clean.

Thus actual development consumed six service startups across all attempts,
not only the three in the valid round. The final r3 was launched as a detached
session so an SSH channel loss could not invalidate its shared in-memory key.

Every valid arm performed scoped TERM-first cleanup with a 60-second grace
period. Each started with six live processes, sent TERM, reached zero live
processes and required no SIGKILL. Per-arm and final postflight found no API
server, worker or GPU process. Fatal/OOM/segfault/timeout/distributed-reset/
worker-loss counts were zero for all arms.

## Layered conclusions

- Experiment validity: `pass` for r3; identities, request populations,
  dispatch, finite values, lifecycle and evidence contracts are complete.
- Operator numerics: retained M1-176 G2 `pass`; not rerun and not contradicted.
- Distribution: both fused-off to M1-109 and M1-109 to M1-162 are
  `distribution_drift_requires_adjudication / inconclusive` relative to the
  exact M1-109 A/A envelope.
- Capability: all three 60-pair development screens have zero discordance, but
  every stratum is underpowered; overall `inconclusive`, not promotion pass.
- Performance: M1-109 to M1-162 mean paired TTFT gain is +2.2763% in six
  diagnostic samples; historical M1-176/M1-177 performance remains separate.
- Promotion: retain both candidates for reviewer consideration. M1-162 remains
  blocked from production promotion; M1-109 is not production-qualified.
  No formal evaluation or default change is authorized by M1-180.

Work stops here for reviewer review.
