# M1-49: Qwen hybrid KV layer accounting

Status: implementation and local gates passed; fixed TP4 legacy/candidate
service qualification is blocked because GPU0 still times out after the SSH
instance reconnect; private branch only; no YAML or `main` change.

## Root cause

Qwen3.6-35B-A3B has 40 decoder layers: ten `full_attention` layers and 30
Gated DeltaNet `linear_attention` layers. The custom Transformers adapter kept
that ownership only in nested `text_config.layer_types`. CoreX vLLM 0.6.3
looks for top-level `hf_config.layers_block_type`; when absent it treats every
hidden layer as attention. CacheEngine therefore allocated 40 KV layers even
though `Qwen3_5Model.forward` consumes a dense list of only ten KV caches.

The old production startup reported 16,878 GPU blocks and 6,553 CPU blocks per
rank. Correcting the per-block byte count from 40 to ten layers should increase
both capacities by approximately four times without adding memory:

| Item | Legacy | Candidate estimate |
| --- | ---: | ---: |
| Attention cache layers | 40 | 10 |
| GPU blocks | 16,878 | 67,512 |
| GPU token capacity | 270,048 | 1,080,192 |
| CPU blocks | 6,553 | 26,212 |

Actual block counts must come from the corrected service startup; estimates do
not qualify capacity or score.

## Fixed implementation

`BI100_HYBRID_KV_ACCOUNTING=legacy40|full_attention` is a private A/B selector.
It defaults to `legacy40`, rejects every other value, and remains absent from
`computility-run.yaml`. Both modes run the same code and immutable evaluator
command.

The selected mode is serialized as `bi100_hybrid_kv_accounting_mode` together
with the derived `layers_block_type`. Reload without an environment override
preserves the serialized mode; a conflicting environment override or stale
serialized layer list fails closed. This prevents worker/config reloads from
silently changing an A/B arm.

In `full_attention` mode the adapter maps nested `full_attention` entries to
top-level `attention`, while preserving `linear_attention` entries. The vLLM
profiling dummy cache list is also changed from total hidden layers to
`get_num_attention_layers`; real CacheEngine allocation and startup profiling
therefore use the same cache count. Model forward fails closed if a non-profile
execution supplies a different count.

The first unguarded smoke exposed the profiling mismatch before this second
fix: vLLM passed 40 empty profiling placeholders and the new exact-count
invariant rejected them after consuming ten. No request ran and no OOM or GPU
fault occurred. The corrected profiler now creates ten placeholders in
candidate mode and 40 in legacy mode. The model separately validates the
configured allocation count, so the legal legacy list of 40 caches is accepted
even though only its first ten dense full-attention ordinals are consumed.

## Fixed gates

1. Local unit discovery and submission preflight must pass; the selector must
   remain absent from YAML and invalid values must fail before model load.
2. On the same installed runtime, legacy startup must report 40 attention
   cache layers and candidate startup ten. Candidate GPU and CPU block counts
   must increase without changing the 0.9 memory utilization or 4-GiB swap
   budget.
3. The existing 65,536 plus two 135,040-token pressure sequence must preserve
   every response digest. Its 20,976 blocks are below the expected candidate
   GPU capacity, so candidate post-pressure should remain GPU-warm and must not
   claim a block-major transfer benefit.
4. Immediate warm latency may regress by at most 2%; Output TPS P10 must remain
   at least 20 and regress by at most 2% in any decode qualification.
5. Fresh 235K warm repeat and 262,144 capacity/exactness must pass without OOM,
   worker loss, collective failure, or loss of multimodal behavior.
6. Only a complete single-session 881 replay with privacy-safe v4 trace may
   determine hit rate, TTFT P90, throughput, and weighted score. Aggregate hit
   multipliers are forbidden.

The M1-46 block-major path is not promoted by this result. Its old paged model
denominator moved 40 unused cache layers, while its data-plane probe used the
ten cache layers consumed by the model. After M1-49, the fixed M1-46 pressure
sequence cannot trigger CPU offload, so continuing its layout A/B would measure
an unexecuted path. M1-46 stops rather than changing the pressure protocol or
scanning request counts.

## Fixed executable A/B

`scripts/install_bi100_bare_host_runtime.sh` first copies the vendor vLLM and
offline Transformers wheel into an isolated site-packages staging directory.
It applies the Docker-equivalent Transformers/CoreX/vLLM patch stack only to
that overlay, verifies package roots and source hashes, then publishes the
runtime with one same-filesystem rename. A failed install deletes staging and
cannot leave system site-packages half-patched. This is required after an
instance restart because `/root` experiment directories are ephemeral even
though the public-storage model persists.

`scripts/run_m1_49_hybrid_kv_ab.sh` encodes the first GPU qualification as a
single fail-closed run. It first requires independent 1,024-square CUDA
matmuls on GPUs 0-3, then starts two fresh TP4 service lifetimes in the fixed
order `legacy40` followed by `full_attention`. Both arms use the same source,
model, evaluator command, `admission64/direct` GDN policy, request salt, and
pressure sequence. The content CPU KV tier, cache tracing, and M1-47 fused
prefill candidate are explicitly disabled. The only service-side difference
is `BI100_HYBRID_KV_ACCOUNTING`.

Each startup is independently checked against the model's real
`AutoConfig`: the legacy arm must expose 40 attention cache layers and the
candidate ten. Every TP worker must also emit its exact rank, environment and
serialized modes, configured KV-layer count, and the ten full-attention layer
ordinals. The rank set must be exactly `0,1,2,3`, with one report per rank.
The safe startup record includes rank-local KV geometry, expected bytes per block, the
final GPU/CPU block counts, logical context length, canonical launch-contract
digests with and without the accounting field, and service-log SHA-256, but no
prompt or model output. The comparator directly requires every startup/runtime
invariant to match after replacing only that accounting field and
recomputes all capacity identities and requires both GPU and CPU block counts
to increase by at least `3.5x`; this is a gate around the expected `4x`
result, not permission to tune memory utilization.

Both arms then run the frozen 65,536-token target plus two 135,040-token
pressure requests with CPU offload disabled. Legacy must retain at most one
16-token target block after pressure, while candidate must retain at least
65,504 tokens. All corresponding response lengths, finish reasons, and
SHA-256 message digests must match. Candidate immediate-warm latency may
regress by at most 2%. Each service runs in an isolated process group; cleanup
kills the whole group, verifies it is empty, waits for port 8000 to be free,
and reruns all four independent GPU probes between arms. Device order, model,
capability, total memory, timeout, matmul size, and deterministic checksum must
match across all three preflights; free memory may change. Cleanup failure
overrides any prior success return code. The harness persists
preflight, startup, per-request, fatal-scan, comparison, and return-code files
to diagnose an interrupted run.

This pressure A/B qualifies layer accounting and resident GPU capacity only.
It does not qualify M1-45 CPU transfer, Output TPS, the complete workload, or
the final score. Fresh 235K/262K and multimodal gates remain locked until the
A/B comparison succeeds.

## Validation so far

- local unit discovery: 219 passed, 13 skipped;
- submission preflight: 8/8 passed;
- remote selector smoke: `legacy40=40`, `full_attention=10`;
- remote patched profiler source uses `get_num_attention_layers`;
- real Transformers save/reload preserves `full_attention/10` without the
  environment, while a conflicting `legacy40` override fails with exit 1;
- installed model helper accepts both configured 40-cache legacy and 10-cache
  candidate contracts;
- atomic bare-host overlay installer, fixed A/B startup/pressure gate, safe
  comparators, and 28 focused unit tests pass locally;
- first unguarded startup rejection was isolated to the 40-placeholder profile
  contract and did not reach benchmark requests.

The latest post-reconnect 1,024-square preflight still returns timeout rc 124
on GPU0; GPU1-3 report `Iluvatar BI-V100`, capability `7.0`, total memory
`34,057,748,480`, and checksum `1,073,741,824.0`. This confirms SSH recovery
did not recover GPU0. No TP4 service may start until all four probes pass.
Historical reset diagnostics and the latest event are in
`evidence/M1_49_RUNTIME_STATUS.json`.

The branch remains private and unqualified until the TP4 service gates above
produce hash-pinned evidence.
