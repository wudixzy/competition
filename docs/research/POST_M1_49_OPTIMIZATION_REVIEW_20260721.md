# Post-M1-49 optimization review

Status: frozen decision record for the private experiment branches. It does
not authorize a YAML or `main` change.

## Plan audit

The original cache-correctness plan is substantially implemented in M1-31
through M1-45:

- full KV blocks use chained SHA-256 content identities rather than physical
  block numbers;
- the first block includes model, adapter, tensor-parallel, dtype, block-size,
  and canonical multimodal identity;
- scheduler metadata carries explicit GDN restore, capture, and eviction
  actions, and workers fail if a selected state is absent;
- effective cached tokens are the contiguous KV/GDN intersection;
- `admission64` keeps at most two useful states per sequence and no longer
  captures every 8,192-token chunk;
- privacy-safe v4 tracing and per-request residual-prefill simulation exist;
- physical-block reuse, pooled forks, multimodal isolation, TP4 pressure,
  131K exactness, 235K stability, and 262K capacity have real evidence.

This means a review based on the early physical-block-key implementation is
obsolete. The missing cache evidence is the complete single-session 881
trace, not another keying redesign.

M1-49 exposes a separate correctness and capacity defect: vLLM 0.6.3 counted
all 40 Qwen layers as KV-owning attention layers even though only ten are
`full_attention`. Correcting this should make the existing content cache much
less eviction-prone. It cannot be converted into a score claim until startup
and request evidence report the actual block counts and reuse pattern.

## External design check

The current vLLM automatic-prefix-cache design hashes a parent digest, the
current block's tokens, and extra identities such as LoRA and multimodal
content; it caches complete blocks and recommends SHA-256 where collision
isolation matters. That matches the identity semantics already backported in
M1-31/M1-45:

https://docs.vllm.ai/en/stable/design/prefix_caching/

Current vLLM's hybrid cache coordinator asks each cache type for a compatible
hit and monotonically reconciles them to one aligned token length. Its full
attention manager first finds a contiguous run of chained full-block hashes,
then optionally probes a finer aligned tail. Porting the V1 coordinator itself
would pull in incompatible scheduler, block-pool, and cache-spec contracts;
the useful invariant for this v0.6.3 backport is the one already enforced:
only the longest content-identical boundary with both live KV and a restorable
GDN state may count as computed.

https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_coordinator.py

https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py

Marconi confirms the sparse-admission rationale. Recurrent state cannot be
trimmed like attention KV, so it admits branch states discovered through
reuse plus final conversation states, with at most two states per sequence.
For chunked recurrence it checkpoints the prior valid chunk boundary rather
than saving every fine-grained state. This matches `admission64` conceptually.
Its decode-end state policy remains locked here because no complete 881 replay
has shown the predeclared additional five-point effective-hit gain.

https://proceedings.mlsys.org/paper_files/paper/2025/file/7c180af017258d239bac6248d1eb26ac-Paper-Conference.pdf

## Why attention is not reopened

M1-47's production-shape kernel improved the fixed 74K/128K/235K core path by
`2.5530x/2.5451x/2.5770x`, remained finite within the fixed `1e-5` relative-L2
boundary, and dispatched on every TP rank. Nevertheless, cold service latency
improved only `3.906%` at 65K and `8.832%` at 235K. This failed the frozen 20%
service gate and closes both the primary fusion and its one allowed structural
alternative. Reopening tile or launch scans would violate the stopping rule.

The old `68.788% layer.full_attn` measurement was inclusive: it did not prove
that the paged-prefix loop accelerated by M1-47 occupied that share. M1-48 is
therefore the next measurement, not a new optimization. Its corrected protocol
uses paired profile-off/profile-on services, streaming TTFT, exact TP-rank and
per-chunk event counts, per-forward rank-spread rejection, and separate
KV-write/dense/paged regions. The old
`2.5770x` microbenchmark is not extrapolated across production chunks. Profile
perturbation above 15% invalidates the report instead of being interpreted.

## Fixed decision sequence

1. Run the M1-49 same-runtime A/B. Reject if four-card preflight, 40/10 layer
   contracts, `3.5x` minimum capacity gain, pressure-output identity, or 2%
   immediate-warm bound fails.
2. Only after that passes, run candidate 235K warm-repeat, 262K exact-capacity,
   and multimodal gates. Capacity estimates never substitute for these runs.
3. Run M1-48 once. Its qualification only authorizes path ranking and always
   records `promotion_authorized=false`. A new cold-prefill experiment is
   allowed only for the largest exclusive region with at least 20% unprofiled
   service headroom. It gets one primary implementation, at most one structural
   alternative, and the same 20% end-to-end cold-TTFT gate.
4. Obtain one complete privacy-safe 881 trace. Use per-request residual
   prefill and measured transfer costs to compare corrected `fine32`,
   `admission64`, and the frequency evictor. Aggregate hit-rate scaling is
   forbidden.
5. Promote nothing until Output TPS P10 >=20, TTFT P90 <=5 seconds, effective
   hit >=50%, success >=99%, weighted score >=8000, and 262144 capacity all
   hold on the fixed evaluator contract.

With the current evidence, no additional YAML parameter, cache capacity,
restore alignment, attention tile, or launch threshold is an admissible
optimization variable.
