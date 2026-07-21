# M1-45: Content-addressed CPU KV tier

Status: allocator metadata, single-GPU data-plane, TP4 eviction-pressure,
131K direct equality, pooled fork/release, and multimodal isolation gates
qualified; 235K/262K and full performance gates pending; disabled in
submission configuration.

## Decision

M1-44 established that the production BI100/CoreX paged-KV transfer path is
bit-exact and meets the fixed transfer limits through 131,072 tokens. M1-45
therefore tests a scheduler-owned, inclusive CPU cache for immutable prefix KV
blocks. It does not alter the model, GDN state format, attention kernels, fixed
competition command, or the existing 4 GiB CPU swap allocation.

The experimental selector is `BI100_CPU_KV_OFFLOAD=0|1`. Its default is `0`,
invalid values fail during allocator construction, and it must remain absent
from `computility-run.yaml` until all qualification gates pass.

## Runtime contract

1. The chained 32-byte SHA-256 block digest is the only content identity.
2. The GPU content map is queried before the CPU map.
3. GPU eviction lazily stages `GPU -> CPU` only when the digest has no existing
   CPU copy. A CPU hit stages `CPU -> GPU` while retaining the CPU copy.
4. A CPU slot used as an H2D source or D2H destination is pinned for the whole
   scheduler step. Pending D2H content cannot be read until the next step.
   D2H may consume free CPU slots immediately, but full-capacity replacement is
   deferred until drain. If the step contains any H2D claim, those deferred
   stores are dropped instead of replacing resident CPU content. This remains
   safe even when D2H scheduling precedes later H2D lookup and preserves all
   not-yet-claimed blocks of a saturated sequential prefix.
5. A reused GPU slot may be a D2H source and then an H2D destination in the same
   step. Every worker must execute all D2H transfers before any H2D transfer.
6. Mappings have unique sources and destinations and are produced only by the
   scheduler. Tensor-parallel workers execute identical explicit mappings.
7. Prefix accounting stops at the first non-computed KV block. The effective
   cache hit remains the intersection of this contiguous KV prefix and a live
   scheduler-owned GDN restore key.
8. The content tier and request-level preemption swap cannot share CPU slots.
   With the fixed `max_num_seqs=1` contract, preemption remains recomputation;
   any attempt to use request-level swap while the tier is enabled fails fast.

## Predeclared gates

- Unit lifecycle: deterministic 10,000-step oracle covering insert, inclusive
  hit, LRU replacement, slot pinning, duplicate maps, pending visibility,
  malformed hashes, and environment validation.
- Installation: Docker copies both allocator modules, `patch_ops.sh` installs
  them, and an idempotent worker patch proves D2H precedes H2D.
- Single GPU: cold/warm branching-prefix replay is finite and deterministic;
  the warm result matches the no-offload reference and logs at least one H2D
  promotion and one lazy D2H preservation without timeout or OOM.
- Single-GPU transfer order: one fixed paged-KV case preserves a victim with
  D2H and restores different requested content with H2D through that same GPU
  slot; the victim, requested value, and inclusive CPU source are bit-exact.
- TP4 correctness: direct mode retains the existing 131K/256-token equality
  boundary; aligned mode passes 235K/1000-token full equality; 256K capacity is
  unchanged.
- TP4 eviction pressure: run the same fixed request sequence after separate
  clean server starts. The sequence is a 65,536-token cold target, its immediate
  warm replay, the minimum number of fixed-length unique pressure requests
  required to exceed the log-reported GPU KV capacity, then two target replays.
  The `control` run must report at most one cached block after pressure; the
  `candidate` run must restore all but at most two target blocks. Every target
  response must have the same completion length, finish reason, and message
  digest. Raw prompts and responses are never persisted.
- Performance: fixed-order A/B, same image and command, cache cleared between
  groups. Admission requires at least +5 percentage points effective hit and
  +5% weighted proxy, Output TPS P10 at least 20 and no more than 2% relative
  regression. Final publication still requires score at least 8000, TTFT P90
  at most 5 seconds, success at least 99%, and effective hit at least 50%.

Failure of a gate keeps the selector out of YAML and `main`; it does not trigger
capacity, threshold, or launch-parameter scanning.

## Allocator gate result

Commit `3e2be3f` was installed into the real CoreX vLLM package on the a163
instance. The fixed allocator lifecycle ran once and qualified:

- initial allocation emitted no transfers;
- eviction emitted D2H `(GPU 1 -> CPU 0)`;
- the next content hit emitted H2D `(CPU 0 -> GPU 1)` while preserving its new
  victim with D2H `(GPU 1 -> CPU 1)`;
- both restored block objects and the allocator's contiguous prefix were
  computed `[0, 1]`;
- `BI100_CPU_KV_OFFLOAD=0` returned empty maps and an invalid value failed at
  allocator construction.

Evidence: `evidence/M1_45_CPU_KV_ALLOCATOR_GATE.json`. This gate validates
metadata and mapping ownership only. It does not qualify CoreX transfer order,
model output, performance, TP4 behavior, or the submission selector.

## Same-slot data-plane result

Commit `1582eb9` ran the fixed same-slot operation once on physical GPU 0 after
all four BI-V100 devices passed independent bounded CUDA smoke tests. The test
preserved GPU victim slot 1 into CPU slot 1, then restored different requested
content from CPU slot 0 into that same GPU slot 1. The preserved victim,
restored request, and inclusive CPU source were all bit-exact. Evidence:
`evidence/M1_45_CPU_KV_SAME_SLOT_ORDER.json`.

This closes the ordering and CoreX data-movement gate. It still does not prove
GDN/KV intersection correctness in the full model, model-output equivalence,
TTFT improvement, 881-request gain, or TP4 stability.

## TP4 pressure harness

`tests/cpu_kv_offload_pressure_api.py` implements the predeclared pressure
sequence through the OpenAI-compatible API. It constructs prompts with exact
token counts using the local model tokenizer, persists progress atomically after
every request, and records only token counts, timings, finish reasons, and
SHA-256 response digests. A failed request, non-finite timing, imprecise prompt,
zero-token completion, changed target response, or missed cache threshold makes
the gate fail closed.

The pressure request count is derived once from the startup log's GPU block
count and block size; it is not a tuning parameter. The same count, token
lengths, run identifier, model command, and request order must be used for the
clean-start control and candidate runs. Candidate evidence is admissible only
when the control proves that pressure actually evicted the target from GPU KV.

## TP4 capacity diagnostic

The diagnostic candidate at runtime commit `132718b` reached healthy status on
all four workers with the fixed 262,144-token command. `/health` and
`/v1/models` returned HTTP 200, and a one-token smoke request completed with
`finish_reason=length`. Startup reported 16,878 GPU blocks and 6,553 CPU blocks
at block size 16, corresponding to 270,048 and 104,848 token slots.

Harness commit `f4aa85c` constructed the fixed prompts with the remote local
tokenizer before any pressure run:

- target: exactly 65,536 tokens, 376,731 UTF-8 bytes, 2.082 seconds;
- pressure: exactly 135,040 tokens, 776,331 UTF-8 bytes, 7.589 seconds.

Two pressure prompts occupy 16,880 blocks, exceeding measured GPU capacity by
exactly two blocks. This is the frozen pressure geometry for both clean-start
runs. The diagnostic is not qualification evidence because it was candidate
first, enabled cache tracing, and predates the saturated-promotion protection.

## Offline replay contract

`scripts/analyze_prefix_cache_trace.py --cpu-capacity-blocks 6553` models the
measured GPU/CPU hierarchy without changing its default zero-CPU behavior. It
uses GPU-first lookup, same-step D2H pending visibility, inclusive CPU copies,
deferred saturated replacement, GDN/KV intersection, and separate prompt and
decode transfer costs. The default per-block costs come from the fixed M1-44
131,072-token medians; prompt transfers extend projected TTFT while decode D2H
extends only projected request latency.

The report contains both zero-CPU control metrics and candidate metrics. Score
projection uses each request's residual prefill and projected latency, never an
aggregate cache-hit multiplier. Reports remain non-qualifying unless the input
is structurally complete at exactly 881 requests and the operator explicitly
passes `--qualification-trace`. No complete real 881-request v4 trace is
currently present in the repository, so synthetic replay cannot qualify this
candidate.

## Formal TP4 A/B

The fixed-order control ran from an isolated experiment directory against the
CoreX-installed package at runtime commit `eef4e1c`. The `/usr/local/corex`
import path and the patch log's resolved `/usr/local/corex-3.2.3` path were
verified with `samefile`, and the installed content-cache source matched the
candidate SHA-256. Launching a manual test from the repository root is invalid
because the checkout's `vllm` package shadows the Docker-equivalent CoreX
installation.

Control A used `BI100_CPU_KV_OFFLOAD=0`, `admission64/direct`, trace disabled,
and fixed run ID `m145-fixed-ab-20260721-16878`:

| Request | Cached tokens | Elapsed | Response digest |
| --- | ---: | ---: | --- |
| target cold | 0 | 92.835 s | `74ac9290bd6f...` |
| target immediate warm | 65,520 | 3.146 s | `74ac9290bd6f...` |
| pressure 0 | 0 | 242.387 s | `d06d9c20276b...` |
| pressure 1 | 16 | 242.660 s | `d06d9c20276b...` |
| target after pressure | 16 | 93.696 s | `74ac9290bd6f...` |
| target refreshed | 65,520 | 3.289 s | `74ac9290bd6f...` |

The control gate qualified, proving that the frozen pressure geometry evicts
the target from GPU KV while preserving deterministic model output. Health
remained HTTP 200 and the fatal/OOM/traceback/worker-loss scan was empty.
Evidence: `evidence/M1_45_TP4_CONTROL_PRESSURE.json`.

Candidate B used the same runtime, isolated launch directory, server command,
run ID, request sequence, and cache geometry. The only experimental change was
`BI100_CPU_KV_OFFLOAD=1`:

| Request | Cached tokens | Elapsed | Response digest |
| --- | ---: | ---: | --- |
| target cold | 0 | 88.890 s | `74ac9290bd6f...` |
| target immediate warm | 65,520 | 3.199 s | `74ac9290bd6f...` |
| pressure 0 | 0 | 233.088 s | `d06d9c20276b...` |
| pressure 1 | 16 | 237.039 s | `d06d9c20276b...` |
| target after pressure | 65,520 | 11.584 s | `74ac9290bd6f...` |
| target refreshed | 65,520 | 3.267 s | `74ac9290bd6f...` |

The strict A/B comparator qualified. After pressure, the candidate retained
65,504 more cached tokens and reduced elapsed time by 82.112 seconds, from
93.696 to 11.584 seconds (`0.1236x`, or 87.64% lower). Every corresponding
completion length, finish reason, and response digest matched. Immediate warm
latency changed from 3.146 to 3.199 seconds (`+1.69%`), inside the fixed 2%
regression limit; the other four comparable requests were faster. Both runs
remained healthy and had empty fatal/OOM/traceback/worker-loss scans.

Evidence:

- `evidence/M1_45_TP4_CANDIDATE_PRESSURE.json`;
- `evidence/M1_45_TP4_PRESSURE_AB.json`.

This qualifies the fixed eviction-pressure correctness gate, not the 881-request
performance gate. The selector remains absent from `computility-run.yaml` and
must still pass 131K direct equality, 235K stability, 256K capacity, and the
predeclared score/TTFT/throughput thresholds before promotion.

## Post-gate namespace audit

An independent review found that `PrefixCachingBlockAllocator.fork()` rebuilt
the first forked block through the raw block pool. That path did not carry the
request cache namespace, so `n > 1` or beam-style sequence forks could derive a
different first-block hash and later fail the cached-block release assertion.

The first fix routed forks through the namespace-aware initializer, but the
real CoreX gate correctly failed both release orders: hashes happened to match
while fork namespaces remained empty. This exposed a deeper object-pool issue.
`BlockPool.init_block()` reinitializes a pre-created block directly and bypasses
the allocator factory, so temporarily setting the allocator namespace did not
affect normal pooled objects. The corrected initializer now restores the
resolved namespace on the returned pooled block, before any content hash can be
observed. The behavioral unit test models that namespace-clearing pool reuse,
then verifies that a two-block fork retains both namespaces and the complete
SHA-256 chain.

The formal pressure and long-context requests are text-only and use one fixed
model with `n=1`, so the empty namespace does not invalidate their timing or
same-run equality evidence. It does invalidate claims that model/adapter and
multimodal namespace isolation were already exercised in the pooled runtime.
The corrected runtime must pass the real CoreX fork/release gate plus same-image
hit and different-image isolation before any evaluation candidate is published.

Runtime commit `4621f0b` then passed the real installed-CoreX gate in both
release orders. Source-first and fork-first each retained two identical chained
hashes and identical namespaces, and all eight blocks returned to the free
pool. The gate reported `qualified=true` with no traceback. Evidence:
`evidence/M1_45_PREFIX_NAMESPACE_FORK_GATE.json`.

This closes the pooled fork/release defect. A standalone API-level
same-image-hit/different-image-isolation gate was still required because the
allocator test does not parse or hash real multimodal request content.

## Multimodal isolation result

The corrected runtime processed three real API requests. The red-image cold
request reported zero cached tokens; its exact replay restored 5,056 tokens,
reduced elapsed time from 28.552 to 1.481 seconds, and retained the same
completion length, finish reason, semantic color, and response SHA-256. The
green-image request used the same text but reported zero cached tokens and
returned the correct different color. This qualifies same-image reuse and
different-image namespace isolation.

The first harness also required red and green images to produce equal prompt
token counts. The runtime legitimately reported 5,070 and 5,072; content-aware
vision processing is not required to assign equal token counts to different
images, and this condition is unrelated to cache isolation. The original
fail-closed report is retained unchanged in
`evidence/M1_45_MULTIMODAL_ISOLATION_RAW.json`. The independent v2 qualifier
requires equal tokenization only for the identical red cold/warm pair, records
the cross-image delta as informational, and qualifies the required behavior in
`evidence/M1_45_MULTIMODAL_ISOLATION_GATE.json`.

## TP4 131K direct equality

The fixed direct-mode run used a 131,000-token prompt and exactly 256 generated
tokens. The cold request took 273.172 seconds with zero cached tokens. Its warm
replay restored 130,992 tokens and took 56.339 seconds. Both responses ended by
the length limit and had the same completion count and message SHA-256 digest
`6cf3195ef4e3...`. Evidence:
`evidence/M1_45_TP4_131K_DIRECT_EXACT.json`.

The result qualifies the predeclared 131K/direct equality boundary. It does not
replace the fresh 235K warm-repeat stability or 262K capacity gates, and it
predates the pooled namespace correction. Its text-only `n=1` timing and digest
remain valid for the reason documented in the namespace audit.
