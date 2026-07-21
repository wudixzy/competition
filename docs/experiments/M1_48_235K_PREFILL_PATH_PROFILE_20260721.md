# M1-48: Post-M1-49 235K prefill path profile

Status: corrected measurement implementation complete; M1-49 prerequisite and
TP4 runtime evidence pending. This private preparation branch is based on
M1-49 `9460b71`. It does not change model mathematics, `computility-run.yaml`,
`main`, Docker defaults, or repository visibility.

## Scope

M1-47 accelerated one fixed paged-prefix microbenchmark shape by `2.5770x`, but
its service experiment used a different request protocol and different query
shapes from this profile. That speedup is historical evidence only. M1-48 does
not apply it to every 235K chunk, does not emit an Amdahl promotion claim, and
does not reopen the closed M1-47 tile scan.

M1-48 answers one narrower question after M1-49 has qualified: which exclusive
production path has enough measured unprofiled-TTFT headroom to justify the one
next implementation experiment?

## Instrumentation contract

The profiler is disabled by default. The diagnostic arm uses exactly:

```text
BI100_PROFILE=1
BI100_PROFILE_MODE=event
BI100_PROFILE_INCLUDE_STARTUP=0
BI100_PROFILE_FILTER=model.*,layer.*,full_attn.*,xformers.*,paged_attn.*,moe.*,gdn_prefix.*
```

The control arm changes only `BI100_PROFILE` to `0`. Both arms retain M1-49
`full_attention`, `admission64/direct`, TP4, a 262144 model length, 8192-token
chunks, block size 16, one sequence, no CPU KV tier, and no fused-prefill
candidate.

Each top-level model forward is a profile transaction. An exception discards
that forward's events and counters, so a later successful request cannot absorb
failed-request data. Payloads bind an explicit TP rank and accept only bounded
scalar metadata. Raw tokens, messages, images, output text, content hashes, GDN
keys, block tables, and physical KV identities are not accepted by the event
schema.

The timers separate:

- exclusive model embedding, norms, GDN, full-attention, MoE, and final norm;
- QGKV projection, norm/RoPE, attention, gate, and output projection;
- XFormers KV write, dense prefill, and inclusive paged prefill;
- the strict-prefix PyTorch paged segments and their exact dispatch geometry;
- request-level worker gaps and the profiled/unprofiled streaming TTFTs.

Prefill context is derived from XFormers' host `seq_lens` list. No metadata
`.item()` is added to the timed path.

CUDA events are the correct primitive for asynchronous GPU timing, but this
implementation still synchronizes at each model-forward flush. The paired
profile-off/profile-on service run therefore measures its own perturbation. If
absolute TTFT perturbation exceeds 15%, the evidence is invalid and the next
step is a deferred request-level flush design, not interpretation of the
distorted profile. See the PyTorch CUDA timing guidance:

https://docs.pytorch.org/docs/main/notes/cuda.html

## Fixed execution

1. Require a qualified `bi100-m1-49-long-context-qualification-v1` report and
   all M1-49 cleanup return codes.
2. Build a new atomic runtime overlay. A separate runtime-identity report must
   prove that the current clean revision, install-time source, and installed
   Qwen model, profiler, paged-attention, and XFormers files all byte-match. It
   also re-hashes the active worker and requires the startup-profile guard.
3. Run `scripts/run_m1_48_prefill_profile.sh`. It refuses dirty source and
   existing output directories. Generated `bench_runs/**` evidence is the only
   source-status exclusion, allowing the default M1-49 output to feed M1-48.
4. Run four-GPU preflight before control, after control, and after profile.
   Free GPU memory may drop by at most 1 GiB from the initial stage.
5. Restart the service between arms. Each service runs in an isolated process
   group and must be fully removed before the next preflight.
6. Send the same exact 235000-token, one-output-token, deterministic streaming
   request to each arm. TTFT is the first non-empty SSE output delta. Both
   requests must be cold, successful, and output-identical by SHA-256.
7. Build the path summary and independent privacy-safe qualification record.
   Source revision, runtime reports, startup gates, service reports, logs,
   preflight comparison, profile summary, and pre-qualification cleanup are
   hash-pinned. A final cleanup failure removes any qualified artifact.

## Fail-closed gates

- exactly ranks `0..3`, one selected cold request, and 29 prefill forwards;
- exact geometry `28 * 8192 + 5624 = 235000` with continuous contexts;
- exact per-forward region/event counts for 40 layers, 30 GDN layers, ten full
  attention layers, and 40 MoE layers;
- all 29 chunks use paged prefill because the fixed chunked-prefill builder
  supplies a block table even at context zero; every full chunk has exact
  strict-block segments `8176+16` and the final chunk has `5616+8` on every
  full-attention layer;
- identical metadata, counters, region counts, and forward indices on all TP
  ranks;
- the cold request performs no restore or eviction and captures exactly one GDN
  state in the final chunk, with no earlier per-chunk captures;
- each rank closes exclusive model and full-attention regions within 3%;
- aggregate and every individual model-forward rank spread are at most 10%;
  profiler TTFT perturbation is at most 15%; worker/model spans cannot exceed
  their enclosing profiled TTFT;
- summary TTFT, overhead, output digest, rank spread, and full-attention fields
  are re-bound to the service evidence by the final qualifier; the qualifier
  independently rebuilds the complete summary from the pinned log and service
  reports and requires canonical equality;
- signed unattributed residuals remain visible; negative values are never
  clamped to hide double counting;
- startup `profile_run()` is marked and excluded without skipping the capacity
  probe or enabling `num_gpu_blocks_override`;
- no CUDA fatal, OOM, process loss, Gloo failure, assertion, leaked process
  group, busy port, extra profiled request, or failed GPU preflight.

## Decision rule

A qualified M1-48 report has scope
`post-m1-49-diagnostic-path-ranking-only` and always records
`promotion_authorized=false`. It may rank one next cold-prefill experiment only
when the largest exclusive region has at least 20% of control TTFT in credible
headroom. That experiment still gets one primary design, at most one structural
alternative, numerical parity gates, and a 20% end-to-end cold-TTFT gate.

No M1-48 result alone changes YAML, `main`, the 881-request requirement, or the
final score gates.
