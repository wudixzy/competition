# M1-136 layered quality gate v2

Date: 2026-07-29

Status: implemented and executed on the private experiment branch. M1-136
failed its frozen legacy absolute-error gate and exposed two harness defects;
it did not establish a gross numeric failure or reject M1-109. The
predeclared M1-138 calibrated adjudication is now required. No default, YAML,
`main`, or repository visibility change is authorized.

## Why v1 was too coarse

Floating-point kernels may use different legal reduction orders. PyTorch
explicitly warns that mathematically identical floating-point computations are
not guaranteed to be bitwise identical. FlashAttention therefore validates an
optimized kernel against an upcast reference with bounded error rather than
requiring bitwise output identity. vLLM model tests similarly accept top-N
mutual support when implementations choose different top tokens. MLPerf
separates implementation equivalence from a task-level quality target and runs
accuracy separately from performance.

Primary references:

- https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
- https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
- https://github.com/vllm-project/vllm/blob/main/tests/models/utils.py
- https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc

These practices do not justify ignoring numerical errors. They require using
the right evidence at the right layer.

## Evidence layers

| Layer | Evidence | Decision role |
|---|---|---|
| Identity and protocol | Same model, tokenizer, template, request parameters, response/SSE schema, tools, reasoning, multimodal and structured output | Hard exact gate |
| Cache transparency | Same-arm cold/warm tokens and structured response, logical cache identity, state availability | Hard exact gate; missing or mismatched state fails fast |
| Operator numerics | Candidate and PyTorch fallback on the same real Q/K/V/cache activations | Hard numeric gate: finite, relative L2 <= 1e-5, max abs <= 1e-3 |
| Teacher-forced distribution | Top-k support, teacher-token logprob drift, top-1 agreement and NLL by prompt cluster | Tight-equivalence pass or escalation; not an operator or capability verdict |
| Autoregressive trajectory | Same-arm repeat, cold/warm repeat and cross-arm greedy suffix | Same-arm/cache exact is hard; cross-arm identity is diagnostic |
| Task capability | Paired tools, reasoning, structured output, code/math, multimodal and long-context tasks | Hard statistical noninferiority gate |
| Performance/lifecycle | TP4 TTFT/TPS/cache/success plus fatal, timeout, worker, Gloo and GPU checks | Hard operational gate |

Semantic task scores can never waive a failed same-activation operator
reference or a broken cache/protocol contract. Conversely, a changed greedy
suffix or teacher-forced top-1 is not itself a task failure when the hard
numeric and protocol layers pass.

## Statistical contract

The five long teacher-forced sequences contain correlated token positions.
Their 320 positions are therefore not treated as 320 independent Bernoulli
samples. M1-134 is a fresh-service A/A calibration and was exactly repeatable;
candidate drift outside the frozen tight envelope now triggers adjudication.

Task capability uses paired request or task-item outcomes and a one-sided 95%
noninferiority bound. For a 2 percentage-point margin, 149 paired items with no
baseline-only regression are the minimum zero-regression screen because
`1 - 0.05^(1/149) <= 0.02`. A small stratum may use a predeclared 5-point
screen with at least 59 zero-regression pairs. Continuous scores use a paired
cluster bootstrap with 20,000 fixed-seed resamples. Margins, strata, and sample
selection cannot be changed after candidate results are observed.

The frozen 149-prompt instruction-following surface and TP4 runner are defined
in `M1_137_IFEVAL_POWER149_FUSED_PREFILL_AB_20260730.md`. That surface is one
part of the required capability matrix; it does not stand in for tools,
reasoning, multimodal, code/math, or long-context recall.

## M1-109 status under v2

M1-109 retains its material performance evidence: component median speedup
1.939x and TP4 cold-TTFT gains of 17.70%, 23.38%, 30.36%, and 36.72% at 32K,
65K, 131K, and 235K. Its synthetic operator comparison was finite with maximum
relative L2 6.625e-6 and max abs 2.441e-4. M1-132 found substantial full-model
distribution drift, while M1-134 proved the control path exactly repeatable.

The resulting classification is `distribution-drift-requires-adjudication`:

1. Run the fused output and existing PyTorch fallback on the same real
   activations at fixed 65K and 131K context buckets on every TP rank.
2. Fail immediately on non-finite output, relative L2 above 1e-5, or max abs
   above 1e-3. No task score can override this.
3. If operator shadow passes, execute the predeclared paired capability matrix.
   Cross-arm token differences are reported but are not the quality metric.
4. Keep M1-108 as the conservative exact-output fallback throughout.

The diagnostic is default-off and bounded to two observations per context
bucket per rank. It records only rank, shape, context length, counts and scalar
error maxima under a private `/tmp` directory. It records no prompts, model
outputs, tensor values, token IDs, or credentials.

## TP4 command

After installing an immutable overlay from the exact clean source revision:

```bash
BI100_RUNTIME_SITE_PACKAGES=/path/to/site-packages \
BI100_RUNTIME_INSTALL_REPORT=/path/to/install.json \
scripts/run_m1_136_fused_prefill_shadow.sh \
  ssh-73ca29ba /tmp/m1-136-shadow-SOURCE
```

The runner sends fixed 65K and 131K requests, requires two observations in
each disjoint context interval on every TP rank, then performs scoped service
cleanup, recorded-session recovery, postflight, four-card preflight comparison,
and fatal/timeout scans. A passing shadow result decides only the hard operator
numeric layer; it does not decide task capability or production promotion.

## Executed result

The run at source revision
`71860f6f668008168295967cae4851cdb83ac13a` collected eight finite 65K
observations before the legacy fail-fast stopped the service. Maximum relative
L2 was `7.1011427252343464e-6`. Six observations passed both legacy bounds; two
failed only fixed max absolute error with `0.001953125`.

The run is invalid for promotion for three independent reasons:

- the frozen `0.001` absolute bound failed;
- rank environment variables were unavailable, so all four process reports
  used an unknown rank;
- the runner did not reap the already-quiescent parent before its cleanup
  audit, which observed two transient zombies.

Final postflight nevertheless found no residual API, worker, or GPU process.
The privacy-safe evidence is recorded in
`docs/experiments/evidence/M1_136_REAL_ACTIVATION_SHADOW_20260730/summary.json`.
M1-138 fixes rank resolution and parent reaping and replaces only the
scale-dependent absolute-error adjudication with a frozen FP32/FP16-rounding
baseline. It does not reinterpret M1-136 as a pass.
