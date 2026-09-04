# BI100 inference validation metrics v2

Date: 2026-09-04

Status: governing policy for new candidate design and historical candidate
reassessment. Existing v1 JSON contracts and historical reports remain
immutable evidence. Their runners must be migrated in a separate, tested
change before a v2 result is called machine-qualified.

## Purpose

This policy separates five questions that the previous gate sometimes mixed:

1. Is the experiment itself valid?
2. Did the implementation preserve protocol, indexing and cache state?
3. Is a floating-point kernel numerically credible?
4. Did the model distribution or task capability materially regress?
5. Is the end-to-end performance gain large and reproducible enough?

No result from one layer answers another. In particular, a different generated
suffix is not by itself a numerical failure, and a good task score cannot
waive a non-finite value, invalid cache state or protocol regression.

## Decision states

Every report must use one of four states:

- `pass`: the predeclared evidence satisfies the current layer;
- `fail`: a candidate-attributable hard gate failed;
- `inconclusive`: valid evidence exists but is underpowered or too noisy;
- `invalid`: the experiment contract, runtime identity, request population or
  lifecycle failed, so no candidate conclusion is allowed.

Infrastructure failure is `invalid` unless the same candidate reproducibly
causes it on a healthy host or the failure is directly attributed to candidate
code. A final promotion run containing any fatal, OOM, collective reset,
worker loss or timeout still fails promotion.

## Metric classes

| Class | Primary metrics | Role |
| --- | --- | --- |
| Run validity | source/runtime/artifact identity, complete population, lifecycle, GPU health | Decide whether evidence may be interpreted |
| Protocol and state | HTTP/SSE schema, usage, tools, cache keys, state identity, capacity | Exact hard invariants |
| Operator numerics | finite values, FP32-calibrated error, LSE/state error, indexing | Detect real mathematical or implementation errors |
| Distribution | teacher-forced NLL/logprob, margin-aware top-k flips, next token | Escalate meaningful model-distribution drift |
| Capability | deterministic contracts and paired task non-inferiority | Decide whether behavior remains useful and correct |
| Performance | TTFT, TPOT/ITL, TPS, goodput, memory and profile attribution | Decide whether optimization value is real |

## G0: experiment validity

A run is valid only when all of the following hold:

- exact source, runtime overlay, model, tokenizer, command, environment and
  candidate artifact identities are recorded;
- control and candidate use the same workload order and request semantics;
- the expected request population is complete, including failed requests;
- raw timing samples are retained and startup/warmup treatment is explicit;
- GPU preflight and postflight pass on every participating card;
- run-owned processes are cleaned with TERM, a 45-60 second grace period,
  survivor-only KILL and wait/reap;
- fatal, OOM, segfault, Gloo/NCCL reset, worker-loss and timeout scans are
  complete;
- no private prompt, image, token, activation, credential or model weight is
  committed.

An incomplete platform population such as 631/881 is not a performance
baseline. Throughput and latency calculated only over successful requests may
be reported diagnostically but cannot qualify a candidate.

## G1: protocol, indexing and cache invariants

These remain exact hard gates:

- valid requests receive the expected HTTP class; invalid requests receive the
  expected 4xx without 5xx or service loss;
- SSE framing, response fields, usage, finish reason, tool JSON, structured
  output, reasoning/content separation, stop behavior and sampling semantics
  match the frozen contract;
- tokenizer, chat template, special tokens, input context and requested output
  budget are unchanged;
- tensor shape, dtype, head mapping, causal mask, block-table order, bounds and
  logical token identity are correct;
- cache keys include all required model, adapter and multimodal identities;
- effective cached tokens are the intersection of live KV and recoverable
  recurrent state;
- missing or mismatched state fails before cached tokens are skipped;
- `max_model_len=262144` and the required large completion budget remain
  available.

Valid deterministic functional rows allow zero new candidate failures. The
final platform success requirement remains at least 99%; local fixed valid
request suites should complete at 100%.

### Cache transparency classes

Cache validation depends on whether reuse changes floating-point execution:

- Pure memoization or byte-copy reuse must preserve cached tensor/state bytes
  and deterministic short greedy output exactly.
- Aligned recomputation or deliberate resegmentation must preserve logical
  identity exactly and pass the same-activation numeric gate at the restore
  boundary. The first-token logits and selected token are required; a later
  low-margin suffix divergence triggers distribution analysis rather than
  automatically proving stale state.

For both classes, a wrong content key, wrong state, incorrect skipped-token
count or cross-request/multimodal state leak is an immediate hard failure.
Semantic scoring cannot waive it.

## G2: operator numerical fidelity

### Reference hierarchy

Each floating-point candidate must use the same input tensors and compare to:

1. an FP32 or higher-precision mathematical reference;
2. the existing production low-precision implementation;
3. the FP32 reference rounded to the production output dtype where relevant;
4. a repeated candidate invocation.

Integer metadata, mask decisions, indexing and zero-padding remain exact.

### Default FP16 acceptance rule

For Attention, MoE and GDN outputs, use calibrated error rather than one
universal absolute threshold:

- all reference, baseline and candidate outputs must be finite;
- candidate relative-L2 error versus FP32 must be no more than `2.0x` the
  production FP16 baseline error versus the same FP32 reference;
- candidate maximum-absolute error versus FP32 must be no more than `2.0x`
  the production FP16 baseline error versus the same reference;
- ratio denominators use a predeclared floor of `1e-12`;
- attention LSE relative-L2 must be no greater than the larger of `1e-5` and
  `2.0x` the production baseline's LSE error;
- stateful GDN candidates must additionally compare every emitted recurrent
  checkpoint and final state norm against the shared reference.

Fixed candidate-versus-rounded `rel-L2 <=1e-5` and `max-abs <=1e-3` are no
longer universal hard gates. They remain scale-sensitive diagnostics and may
be retained by an operator-specific contract when justified before observing
candidate results.

The `2.0x` rule follows the role used by FlashAttention tests: optimized
low-precision attention is judged against the error of a conventional
low-precision implementation relative to a higher-precision reference.

### Repeatability

Bitwise repeatability is required only when the declared algorithm and current
production path are deterministic on the same hardware and schedule. A kernel
with documented atomics or scheduling nondeterminism must instead declare a
repeat-to-repeat numeric envelope before testing. Unexplained intermittent
non-finite values, large outliers or input-dependent races are hard failures.

Cross-host, cross-version or cross-binary bit identity is not required. Source,
toolchain, artifact hashes and metric reproduction are required.

## G3: model-distribution fidelity

Teacher-forced evaluation characterizes error propagation; it is not a second
operator error norm and is not a standalone capability verdict.

Every comparison first runs a control/control A/A calibration on the same
runtime. Candidate reports then include:

- top-1 and mutual top-k agreement;
- teacher-token and shared-token logprob deltas;
- paired mean NLL difference with a one-sided 95% confidence interval;
- first divergent greedy position and the control top-1 margin;
- high-margin flips using a threshold frozen from A/A noise, not selected from
  candidate results.

Default review thresholds are:

- high-margin cutoff: `max(0.10 nats, 4 * A/A p99 shared-logprob delta)`;
- zero unexplained high-margin flips;
- candidate mean-NLL regression upper confidence bound no larger than
  `max(0.01 nats, 2 * A/A upper confidence bound)`.

Crossing either threshold is `distribution_drift_requires_adjudication`, not an
automatic numeric failure. It blocks promotion until expanded task evaluation
or root-cause analysis resolves it. Overall top-1 agreement is diagnostic and
has no universal 98% hard threshold because near-tied tokens can flip without
material probability or capability change.

## G4: capability and behavior

### Deterministic contracts

Protocol, tools, structured output, stop sequences, multilingual encoding,
multimodal isolation and explicitly checkable long-context facts permit zero
new baseline-only failures. These are contracts, not statistical model-quality
samples.

### Statistical capability suites

Use paired examples and report both-pass, baseline-only, candidate-only and
both-fail counts. The default rules are:

- development screen: one-sided 95% paired lower confidence bound above a
  `-5` percentage-point margin; this only authorizes expanded testing;
- promotion suite: one-sided 95% paired lower confidence bound above a `-2`
  percentage-point aggregate margin;
- report an exact paired McNemar diagnostic alongside the bootstrap interval;
- mark underpowered results `inconclusive`, never pass or fail by convenience;
- report code, reasoning, tools, structured output, multimodal and long-context
  strata separately; an aggregate gain cannot hide a critical-stratum failure.

The two- and five-point margins are project risk limits, not universal model
constants. They must be frozen with dataset revision, sample size, selection
rule and evaluator before candidate outputs are observed.

Exact output text is required only where the task contract defines an exact
answer. For open generation, task correctness and paired non-inferiority carry
the capability decision.

## G5: performance and resource metrics

### Required reporting

Every service candidate reports:

- TTFT, TPOT/ITL and end-to-end latency at P50/P90/P99;
- Input TPS, Output TPS, Cache TPS and request throughput;
- SLO goodput for the active TTFT/TPOT targets;
- success/error rate over the complete attempted population;
- cold, partial-prefix and warm cache results;
- input/output-length buckets, including 4K, 16K, 32K, 64K, 131K, 235K and
  near-262K where the stage requires them;
- peak device memory, KV capacity, startup/model-load time and cleanup state;
- operator and end-to-end timings as separate claims.

### Component continuation rule

Replace a universal `1.5x` component speed floor with a profile-based Amdahl
screen. Before implementation, record hotspot fraction `f`, candidate speedup
`s`, and projected end-to-end gain:

`projected_gain = 1 / (1 - f + f / s) - 1`

A component continues when the conservative projected end-to-end gain is at
least 2%, or when it directly addresses a currently failed final hard metric
such as long-context Output TPS. Rare-shape regressions are weighted by the
frozen workload; common request buckets still require explicit protection.

### Measurement rule

- Operator screens use at least five post-warmup interleaved timing samples per
  cell and report raw values, median and dispersion.
- TP4 screens use order-balanced paired arms. Start with three pairs for a
  large expected effect; add at most two preauthorized pairs when the estimate
  is useful but its interval remains inconclusive.
- A private candidate normally advances when the paired end-to-end estimate is
  at least 3% and its one-sided 95% lower bound is above zero.
- A point estimate below 2% stops the direction unless it fixes correctness or
  a final hard constraint. The 2%-3% interval is `inconclusive` and should be
  resolved with the bounded extra pairs, not parameter scanning.
- No common workload bucket may show a statistically supported regression
  greater than 5%. A single noisy cell or one platform run is insufficient to
  declare such a regression.

## G6: final promotion

The competition requirements remain unchanged and apply together only to the
exact final TP4 artifact:

- Output TPS P10 at least 20;
- TTFT P90 at most 5 seconds;
- effective cache hit rate at least 50%;
- request success rate at least 99%;
- weighted throughput at least 8000;
- `max_model_len=262144` and required large completion budgets;
- full protocol, tool, reasoning, structured-output, multimodal and
  long-context capability;
- no fatal, OOM, collective reset, worker loss, timeout or dirty postflight.

Passing an earlier layer cannot waive any final requirement. Conversely, final
targets must not be imposed as pass/fail criteria on an isolated L1 kernel.

## Historical candidate impact

- M1-162 remains qualified for real-activation replay under the calibrated
  FP16 rule.
- M1-109 remains open for distribution and capability adjudication; its greedy
  divergence alone is not a failure, while its observed teacher-forced drift
  still requires explanation.
- M1-38 small-batch MoE and M1-19 W13 may each receive one bounded calibrated
  reassessment because their old fixed-error rejection is not sufficient.
- E-ATTN06 remains rejected: its 0.05937 maximum error is a large real numeric
  signal, not a harmless near-tie trajectory difference.
- stale or missing GDN/KV state, false cache hits, protocol field loss and
  non-finite output remain hard failures without reassessment.

## Contract migration

`quality/layered_quality_gate.v1.json` and
`quality/experiment_funnel.v1.json` remain the identities for old evidence.
Do not edit them in place. A later implementation change must add versioned v2
contracts, update validators and unit tests, and prove that historical reports
are still parsed under v1 semantics. Until that change lands, reports should
state both the executed machine contract and this review policy.

## Primary references

- FlashAttention accuracy methodology:
  https://github.com/Dao-AILab/flash-attention
- PyTorch numerical accuracy:
  https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
- vLLM serving and goodput metrics:
  https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py
- vLLM reproducibility:
  https://docs.vllm.ai/en/stable/usage/reproducibility/
- MLPerf inference accuracy and latency rules:
  https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
- TensorRT-LLM serving benchmark metrics:
  https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/run-benchmark-with-trtllm-serve.html
