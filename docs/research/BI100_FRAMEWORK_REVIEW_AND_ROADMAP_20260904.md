# BI100 framework review and optimization roadmap

Date: 2026-09-04

Status: research and planning only. This review does not authorize a formal
`computility-run.yaml` change, a default-selector change, a `main` merge, an
official-score claim, or a repository-visibility change.

Metric selection and candidate decisions follow
`docs/BI100_VALIDATION_METRICS_V2_20260904.md`. Historical v1 evidence keeps
its original interpretation; new candidates use calibrated low-precision
numerics, margin-aware distribution review, paired capability non-inferiority
and profile-based performance continuation rules.

## Executive assessment

The project is no longer blocked on finding any viable optimization. It has a
qualified cache-correctness baseline, a materially faster prefill candidate,
accepted decode-time MoE/GDN/collective improvements, and a layered experiment
funnel. The main problem is that those results are at different evidence and
integration stages:

- formal `main` is still `fb0084fc778e62c26d6a6e108b87dc027ae2ed79`;
- the current private research branch is 80 commits ahead of `main` and touches
  464 files, so it must not be promoted wholesale;
- M1-165/M1-167 fix and validate `max_completion_tokens`, but the fix has not
  yet been validated by a new 881-request platform run;
- M1-109 has strong TP4 prefill evidence and M1-162 adds another 15%-17%
  operator improvement; M1-176 has added TP1-derived real-activation screening,
  but a valid full-model TP4 candidate arm has not yet run;
- long-context decode still gathers the complete K/V state for every generated
  token and is the main unresolved Output TPS bottleneck;
- the latest platform result completed only 631/881 requests and timed out, so
  its latency and throughput distributions are conditional on an incomplete
  request population.

The next cycle should close evidence and integration gaps before creating more
kernel variants.

## Current evidence map

| Area | Best current evidence | Current status | Main blocker |
| --- | --- | --- | --- |
| OpenAI protocol | M1-165 immutable CoreX overlay probe, 10/10 expected cases; M1-167 field interactions, 18/18 | Qualified on private branch | No fresh platform result and not on formal `main` |
| Prefix/GDN cache | M1-107 TP4: 18/18 exact outputs, effective hit rate 49.93% to 62.78%, Output TPS P10 21.986 | Correctness baseline retained | Capture overhead and state management, not raw hit-rate headroom |
| Fused prefill | M1-109 TP4 cold TTFT gains of 17.70%-36.72% at 32K-235K | Mature candidate, default off | Quality adjudication and clean integration |
| FP16-QK prefill | M1-162/M1-176: 15.3%-17.2% synthetic and about 17% on TP1-derived real activations | Development operator screen passed | Valid full-model TP4 service A/B |
| Long decode | E-ATTN05 exact gather is 1.50x-2.02x over the old path | Production fallback | Output TPS falls to 3.698 at 235K |
| Decode MoE/GDN | E-MOE20 direct routed MoE, packed GDN decode and IPC all-reduce have positive TP4 evidence | Existing production stack | Re-attest actual Docker activation and interaction |
| Small-batch MoE | M1-38 reports about 26x-30x at T=2/8/16 | Reopen under layered numeric gate | Real activation, FP32 oracle and model-level quality |
| Long-prefill MoE | M1-19 W13 reports about 1.70x-1.73x on a real route trace | Bounded reassessment only | W2, GPU-only routing and calibrated numerics |
| GDN prefill | Historical 235K profile attributed 16.32% of model time | Profile-dependent opportunity | Profile is stale after M1-109 |
| Experiment funnel | M1-176 L0-L2 implementation and one invalid L3 attempt | Needs simplification | Over-broad gates and fail-open v2 qualifiers |

## Platform result interpretation

The `fb0084f` result is useful for failure attribution but is not a valid final
performance baseline:

- success: 631/881, or 71.62%;
- errors: 250, including 226 tool-related and 22 image-related 4xx;
- Output TPS P10: 3.81;
- Input TPS mean: 1602.62;
- TTFT P90: 30.805 seconds;
- cache hit rate: 0.54;
- workload timeout: true.

M1-168 found that the error population was unchanged from `503fa7c`, so this
does not establish a new model regression. The result predates the
`max_completion_tokens` compatibility fix. A new protocol-compatible platform
run is required before comparing performance percentiles or projecting an
official weighted score.

The global TTFT P90 is primarily controlled by the upper 16K-32K and lower
32K-64K requests. Optimizing only the four 128K-256K requests can materially
improve long-request latency without moving the submitted P90 enough. The
development matrix must therefore keep 16K, 24K, 32K, 48K and 64K cells.

## Structural bottlenecks

### 1. Protocol and result-population integrity

Request validation occurs before tokenization and model execution. A protocol
failure both violates the success-rate gate and removes requests from the TPS
and latency populations. M1-165/M1-167 should be integrated before using a new
platform run as performance evidence. Unknown fields must remain forbidden;
only evidence-backed, lossless mappings should be added.

### 2. Prefill evidence closure

M1-109 has already demonstrated end-to-end value. M1-162 is the strongest
incremental prefill candidate and reproduced across instances. M1-176 completed
the bounded TP1 capture and four independent rank-shaped replays; this is useful
development evidence, but it is not a real TP4 model execution and should not
be described as one.

The attempted L3 run was invalid because it used a reduced diagnostic model and
the second control arm did not finish. The candidate never started. Review also
found that the qualifier accepted insufficiently bound distribution evidence
and ignored the second control in the performance estimate. The next useful
step is to repair those decisions and run a small full-model TP4 A/B, not to
expand provenance checks or rebuild a nine-cell activation matrix.

### 3. Long-context decode

For contexts above 32K, the current fallback gathers the full K/V sequence on
every generated token and performs FP32 QK, softmax and PV as separate steps.
Measured 64-token generation throughput falls from 10.188 TPS at 32K to 3.698
TPS at 235K. This is the clearest structural reason the Output TPS P10 gate is
not met.

The next bounded design should target only the production rank-local shape:
FP16 storage, query heads 4, KV heads 1, head dimension 256, block size 16 and
query length 1. It should load paged K/V directly, reuse one K vector across
four query heads, maintain FP32 online max/sum/output state, and avoid global
score or weight tensors. E-ATTN06 must not be reused unchanged: its 100K
maximum absolute error of 0.05937 is a hard numeric failure.

### 4. Post-attention MoE and GDN

The old 235K profile assigned 67.77% of model time to full attention, 16.32%
to GDN and 14.92% to MoE. Once attention is accelerated by roughly 2x, GDN and
MoE become a much larger fraction of the remaining time. A fresh profile is
therefore mandatory before using the old percentages to reject either area.

M1-38 is the first MoE candidate to reassess because it targets the T=2/8/16
residual prefill sizes common after a prefix hit and has unusually high
component upside. M1-19 is second and should receive only one calibrated W13
retest before W2 or routing work is authorized.

For GDN, checkpoint/state-copy fusion and a preallocated indexed state pool are
lower-risk first steps. A new GDN math kernel is justified only if a post-M1-162
profile still shows a material end-to-end ceiling.

### 5. Cache management

The content-keyed, scheduler-owned hybrid cache is the correctness baseline.
Its 62.78% effective hit rate is already close to the workload's approximately
65.6% theoretical reusable-token ceiling. Further work should reduce capture,
clone and eviction overhead rather than inflate `cached_tokens`.

The final complete-prefill checkpoint must be retained: M1-169's
`tail64_nofinal` policy reduced effective hit rate to 14.68%. A reuse- and
recompute-cost-aware admission policy may be screened offline, but it must
preserve content identity, live-KV intersection and restorable GDN state.

## Ranked roadmap

### R0: Restore trustworthy integration evidence

1. Freeze `main@fb0084fc` as the platform comparison point.
2. Create a clean private integration branch from `main`; do not merge the
   current 80-commit research branch wholesale.
3. Cherry-pick only M1-165/M1-167 protocol support, privacy-safe 4xx
   attribution, their runtime introspection checks and focused tests.
4. Re-run unit, preflight and immutable runtime introspection. Submit one new
   platform build to determine whether success reaches at least 99%.
5. Do not interpret its performance until all expected request classes enter
   serving and unexplained 4xx are zero.

Exit criterion: a complete or fully attributed 881-request population with no
protocol-induced 5xx, worker loss or fatal error.

### R1: Complete M1-162 prefill qualification

1. Fix the v2 fail-open review findings with focused unit tests: bind and
   validate distribution evidence, reject empty validity/capability evidence,
   classify negative performance as a candidate failure, and either use every
   measured control arm or omit it from the design.
2. Add an explicit `attention_operator` change scope. Remove same-host `/tmp`,
   permission, per-file hash, HMAC, clean-tree, repeated preflight and unrelated
   cache/protocol/capability checks from this development path.
3. Retain M1-176 L1 and TP1-derived activation results as an operator screen.
   Head/KV mapping needs one focused unit proof; do not rerun broad reassembly
   and capture matrices for an unchanged mapping.
4. Use the fixed full model for one control and one M1-162 candidate TP4 service.
   Batch 16K, 32K and 64K cold requests, two or three repetitions each, with the
   same request semantics and a small fixed output budget. Require control
   selector absence, candidate dispatch, complete requests and clean cleanup.
5. If gain is at least 5% and buckets are stable, proceed. If it is 2%-5% or
   noisy, run at most one reversed pair or reusable A/A calibration. Below 2%,
   stop this incremental candidate unless profile attribution explains a hard
   metric benefit.
6. Only a surviving candidate runs 131K and 235K cold confirmation, then focused
   teacher-forced drift checks. Cache partial-prefix tests are not part of this
   attention-only path; investigate the prior sibling timeout separately as a
   baseline cache issue.
7. Near-262K capacity and the full capability/protocol/cache suite remain final
   integration gates.

Exit criterion: a valid full-model TP4 candidate dispatch with material TTFT
gain, finite/calibrated numerics and no observed short-request regression. This
authorizes long-context confirmation, not main or YAML promotion.

### R2: Re-profile the accepted stack

Profile the same TP4 runtime with M1-162 active at 16K, 32K, 64K, 131K and
235K. Attribute at least QK, PV, normalize, mask, GDN, MoE, collectives,
scheduler/CPU gaps and kernel-launch/synchronization time. Keep prefill and
decode profiles separate.

Exit criterion: every subsequent implementation names a measured end-to-end
ceiling and a minimum useful gain before coding begins.

### R3: Resolve long-context decode

Implement at most two reasonable direct paged online-softmax designs for the
fixed 4/1/256/16/q1 shape. Screen at 32K, 65K, 131K and 235K using FP32 oracle
calibration, finite checks, repeated determinism and fixed greedy next-token
tests. Only a component-qualified design enters a TP4 64/256/1000-token
generation A/B.

Stop after two designs if neither improves the current E-ATTN05 path by at
least 1.25x at 65K and 131K while passing the numeric layer. Do not scan tile
sizes indefinitely.

### R4: Reassess small-batch MoE

Reopen M1-38 under the layered gate at T=2/8/16. Compare direct and vendor
paths to a shared high-precision oracle, then replay real routed activations.
If qualified, integrate it behind a default-off selector and measure warm
partial-prefix TTFT and Cache TPS. M1-19 receives one W13 calibrated retest
only after M1-38 or the new profile shows a larger long-prefill MoE ceiling.

### R5: Reduce GDN state overhead

First measure checkpoint copy, allocation and capture-boundary split costs.
Then evaluate one implementation that writes up to two admitted checkpoints
into a preallocated state pool from the existing 64-token computation. Preserve
scheduler-owned capture/eviction actions and fail-fast restore semantics.

Do not start a new CoreX GDN math kernel unless the refreshed profile predicts
at least 5% end-to-end improvement after all integration costs.

### R6: Final integration and promotion

Build a minimal candidate branch from the clean protocol branch and cherry-pick
only qualified cache, prefill, decode and MoE/GDN changes. Rebuild one immutable
Docker overlay and run the full protocol, cache-transparency, capability,
long-context, stability and TP4 performance gates on that exact artifact.

Only then may a `main` or formal YAML change be proposed.

## Experiment budget and stop rules

- Route tests by change scope. Operator-only work does not inherit cache, API or
  capability matrices; those gates are added only when touched code can affect
  the corresponding behavior. L4/L5 remain cumulative final-candidate stages.
- Cache build artifacts and overlays by revision/toolchain. Run GPU and NCCL
  preflight once per stable machine session, not before every replay cell.
- A short TP4 operator screen starts with two service startups total: one
  control and one candidate. A/A or an order reversal is conditional on a
  gray-zone/noisy result, not mandatory overhead for every candidate.
- A component speedup is not a service claim. A changed greedy suffix is not
  automatically a numeric or capability failure. A failed same-activation
  numeric comparison, nonfinite value, protocol semantic change or invalid
  cache state remains a hard rejection.
- Component continuation is based on a measured hotspot and at least 2%
  conservative projected end-to-end gain, not a universal 1.5x speedup.
  Short TP4 advancement normally requires at least 3% paired gain with a
  one-sided 95% lower confidence bound above zero; 2%-3% is inconclusive and
  permits only the bounded extra pairs defined by v2.
- Do not rerun M1-173 batched split PV, M1-174 query-tiled scalar fusion,
  M1-172 unsupported mixed PV, blind ixinfer FMHA configurations, scalar GDN
  variants, or YAML/chunk/threshold sweeps without new profile evidence.
- Every full-model runner must own its process group, use TERM with a
  45-60 second grace period before survivor-only KILL, wait/reap children, and
  require clean GPU/process/fatal postflight.
- Keep credentials, model weights and raw private datasets out of Git. Ordinary
  same-host development reports and activations may live in gitignored storage;
  do not add permission, HMAC or per-file checksum gates unless they protect a
  concrete correctness or transfer boundary.

## Promotion gates

The final candidate must independently satisfy:

- request success at least 99%;
- Output TPS P10 at least 20;
- TTFT P90 at most 5 seconds;
- effective cache hit rate at least 50%;
- weighted throughput at least 8000;
- 262144 context capacity and required completion budgets;
- protocol, streaming, tools, reasoning, structured output and multimodal
  behavior;
- cache cold/warm transparency and fail-fast state restoration;
- finite, calibrated custom-kernel numerics and no material task-capability
  regression;
- no fatal, OOM, Gloo/NCCL reset, worker loss, timeout or dirty postflight.

Performance and model capability remain separate conclusions. The 13-request
development set and any incomplete platform run cannot substitute for the full
promotion evidence.

## Reference lineage

- `docs/experiments/M1_139_EFFICIENT_EXPERIMENT_FUNNEL_20260730.md`
- `docs/BI100_VALIDATION_METRICS_V2_20260904.md`
- `docs/experiments/M1_151_TTFT_P90_PREFILL_GRID_20260730.md`
- `docs/experiments/M1_156_FUSED_PREFILL_PHASE_PROFILE_20260730.md`
- `docs/experiments/M1_162_CALIBRATED_FP16_QK_REASSESSMENT_20260730.md`
- `docs/experiments/M1_165_MAX_COMPLETION_TOKENS_COMPAT_20260730.md`
- `docs/experiments/M1_167_CHAT_FIELD_INTERACTIONS_20260731.md`
- `docs/experiments/M1_168_PLATFORM_FB0084F_TRIAGE_20260801.md`
- `docs/experiments/M1_175_FP16_QK_CROSS_INSTANCE_20260804.md`
- vLLM prefix caching design:
  https://docs.vllm.ai/en/stable/design/prefix_caching/
- vLLM hybrid KV cache manager:
  https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/
- Marconi state admission and eviction:
  https://proceedings.mlsys.org/paper_files/paper/2025/hash/7c180af017258d239bac6248d1eb26ac-Abstract-Conference.html
- FlashInfer paged prefill data flow:
  https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/prefill.cuh
