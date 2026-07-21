# M1-49: Qwen hybrid KV layer accounting

Status: implementation and local gates passed; fixed TP4 legacy/candidate
service qualification pending an instance-level GPU0 reset; private branch
only; no YAML or `main` change.

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
candidate mode and 40 in legacy mode.

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

## Validation so far

- local unit discovery: 196 passed, 13 skipped;
- submission preflight: 8/8 passed;
- remote selector smoke: `legacy40=40`, `full_attention=10`;
- remote patched profiler source uses `get_num_attention_layers`;
- first unguarded startup rejection was isolated to the 40-placeholder profile
  contract and did not reach benchmark requests.

After installing the profiler fix, GPU0 failed the independent preflight while
GPU1-3 passed. A 256-square GPU0 retry also timed out after 15 seconds. `ixsmi`
reported 257 MiB, 100% utilization, and no compute process. A device reset was
rejected because host PID 54048 owns the device, but that PID is absent from the
container PID namespace and the `ixsmi` compute-app query. No further TP4
service may start until the instance is reset and all four preflights pass.
Hash-pinned details are in `evidence/M1_49_RUNTIME_STATUS.json`.

The branch remains private and unqualified until the TP4 service gates above
produce hash-pinned evidence.
