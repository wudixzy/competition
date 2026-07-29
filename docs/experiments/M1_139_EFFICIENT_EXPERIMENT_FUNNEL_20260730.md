# M1-139 efficient experiment funnel

Date: 2026-07-30

Status: the funnel contract, capture/replay harness and runners are implemented
on the private experiment branch. Local validation passes. BI100 compile,
overlay and parallel-preflight pilots are complete. L2 capture/replay was not
started because only GPUs 2 and 3 passed the hardware preflight. No formal
YAML, default selector, `main`, or repository visibility change is authorized.

## Audit result

The old process applied final-stage evidence requirements too early.
`run_m1_99_fused_prefill_service_ab.sh` starts six fresh TP4 services and sends
72 requests for one three-pair performance decision. The M1-116 32K
truncation cases alone consumed 1494.667 and 1501.907 seconds. Historical cold
131K, 235K and near-262K requests consumed 273.172, 589.673 and 615.437
seconds. These costs are justified for confirmation, but not for every kernel
edit.

Other repeated work was also avoidable:

- attention-only changes could enter a sequential queue containing unrelated
  MoE and GDN probes;
- the extension build and exact-commit runtime overlay had no reusable
  content-addressed ensure layer;
- most runners did not report monotonic stage wall time;
- M1-136/M1-138 recomputed the same real-activation reference during each
  fresh service run;
- first-observation shadow sampling did not span fixed full-attention layer
  ordinals.

The pilot also exposed four harness defects before a model startup was spent:

- the CoreX builder used the system Python instead of the active CoreX Python;
- its expected artifact name differed from the build script's output;
- overlay verification and several new runner subprocesses were pinned to the
  host's Python 3.8 instead of the active Python 3.10;
- the original four-GPU preflight was serial and emitted no result until all
  probes had finished.

The structured audit is
`evidence/M1_139_EXPERIMENT_FUNNEL_AUDIT_20260730.json`.

## Frozen funnel

`quality/experiment_funnel.v1.json` freezes six stages:

| Stage | Execution | Decision surface |
|---|---|---|
| L0 | CPU static, ABI and unit checks | Reject malformed candidates |
| L1 | Four independent production shapes on four GPUs | Synthetic numeric and kernel-speed screen |
| L2 | One private baseline capture, then four parallel rank replays | Real-activation numeric and kernel-speed screen |
| L3 | One TP4 startup per arm, batching 4K/32K/65K | Dispatch, cache transparency and short integration |
| L4 | 131K/235K/262K TP4 confirmation | Long-context gain, stability and capacity |
| L5 | Complete performance and capability gates | Promotion proposal only |

Every stage authorizes only the next stage. Cache transparency, protocol
semantics, malformed results, nonfinite values and genuine numerical failures
remain hard gates. Semantic scores cannot waive a numeric failure. Conversely,
cross-arm greedy divergence alone is not treated as a capability failure.

## Reusable real activations

The default-off capture path runs only with the fused candidate disabled. It
also requires an explicit `synthetic-exact-prompt-v1` attestation and a private
directory below `/tmp`. For each TP rank it captures fixed context buckets and
full-attention call ordinals 0, 4 and 9. Physical block identities are compacted
and the logical block table is remapped without changing attention order.

Raw tensors are mode 0600, remain outside the repository, and may not be
committed. Manifests contain only shapes, dtypes, sizes and SHA-256 identities.
Candidate commits are deliberately distinct from the capture commit, allowing
one baseline bank to screen many isolated `.so` artifacts.

Replay performs the M1-138 calibrated FP32/FP16-rounding gate, checks LSE,
records reference and candidate kernel timing, and assigns one captured TP
rank to each physical GPU. It cannot validate collectives, end-to-end TTFT,
model capability or production readiness.

## Caches and reports

`build_cached_corex_fused_prefill.py` keys builds by kernel source, build
script, compiler binary/version, Python, Torch identity, ABI and ivcore10
target. Cached artifacts are rehashed before publication.

`ensure_bi100_runtime_overlay.py` reuses only an exact clean Git commit and
still runs full runtime-tree identity verification on every use. It does not
permit cross-revision overlay reuse.

`record_experiment_timeline.py` records privacy-safe monotonic start/end events
and reports wall span, summed stage time and effective parallelism. The new
capture, replay and short TP4 runners retain scoped TERM-first cleanup,
postflight, repeated four-card preflight and fatal-category scans.

## Actual BI100 pilot

The pilot ran on `ssh-73ca29ba` without starting the model:

| Reusable step | First run | Reuse | Measured improvement |
|---|---:|---:|---:|
| CoreX extension compile | 24.647 s | 1.677 s | 14.69x |
| Exact-commit runtime overlay | 8.824 s | 0.303 s | 29.08x |
| Four-card preflight | legacy serial did not finish within 180 s | parallel 89.412 s | at least 2.01x |

The compile miss and hit produced the same 247,176-byte artifact with SHA-256
`f94ad8abb554c6b2eb1a972ad7198cc4d57d36a166afe3e9b869985dda543236`.
The overlay miss and hit both verified runtime-tree SHA-256
`8eadda8dc05cb46d917fb574da13be441b736e5ec57b86ef70a4defc29b0f60b`.
Reuse still performs artifact rehashing or full runtime-tree verification.

The parallel preflight completed and reaped all four scoped process groups.
GPUs 2 and 3 passed allocation and matmul. GPUs 0 and 1 both timed out at
`torch.cuda.mem_get_info()` and required SIGKILL after the full TERM grace
period. Compared with the observed serial external timeout, parallel diagnosis
saved at least 90.588 seconds and returned per-GPU failure stages instead of an
undifferentiated timeout.

This is infrastructure evidence, not candidate performance evidence. The
activation bank, rank replay, short TP4 screen, long-context confirmation and
capability gates remain pending until all four GPUs pass preflight.

## Evidence boundaries

Fast stages retain source revision, build/toolchain key, artifact digest,
runtime-tree digest, input tensor shapes and digests, calibrated numerical
metrics, stage wall time, cleanup status and fatal-category counts. Raw
activations stay mode 0600 under `/tmp` and are not committed.

Only full TP4 runs can establish:

- collective and multiprocess stability on the production topology;
- dispatch correctness inside the complete model;
- cold/warm output and cache transparency;
- 4K through 262K end-to-end TTFT and throughput;
- tool calling, reasoning, multimodal and long-context capability;
- eligibility for L4/L5 promotion, YAML changes or `main`.

## Validation

Current local validation:

- complete unit discovery: 1409 passed, 26 skipped;
- Python syntax checks: pass;
- shell syntax checks: pass;
- Git diff whitespace check: pass.

The BI100 pilot added actual compile/overlay cache and parallel-preflight wall
times to the structured audit. Capture and four-way replay wall times remain
unset rather than being inferred from synthetic or two-card data. Only a
full-profile L2 pass may authorize L3. L4 and L5 remain mandatory before any
production proposal.
