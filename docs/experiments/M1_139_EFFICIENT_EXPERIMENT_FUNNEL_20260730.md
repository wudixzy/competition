# M1-139 efficient experiment funnel

Date: 2026-07-30

Status: the funnel contract, capture/replay harness and runners are implemented
on the private experiment branch. Local validation passes. BI100 pilot timing
is pending. No formal YAML, default selector, `main`, or repository visibility
change is authorized.

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

## Validation

Local validation before the BI100 pilot:

- complete unit discovery: 1399 passed, 26 skipped;
- Python syntax checks: pass;
- shell syntax checks: pass;
- Git diff whitespace check: pass.

The BI100 pilot must add actual cache miss/hit, capture and four-way replay wall
times to the structured audit. Only a full-profile L2 pass may authorize L3.
L4 and L5 remain mandatory before any production proposal.
