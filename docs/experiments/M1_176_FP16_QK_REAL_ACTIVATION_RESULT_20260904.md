# M1-176 FP16-QK real-activation and short-TP4 result

## Decision

M1-176 passed L0, the frozen L1 synthetic screen, and L2 four-rank
real-activation replay. All 12 rank-local L2 cells passed the v2 G2 calibrated
numeric gate. L3 is `invalid`, not a candidate failure: the candidate arm was
never started, the retry used a four-layer diagnostic checkpoint rather than
the required full model, and control B timed out with an incomplete request
population. No TP4 service performance, teacher-forced distribution, or
capability conclusion is authorized.

Work stops here for reviewer adjudication. This run did not enter long-context
confirmation, the formal 881 workload, direct decode, MoE, GDN, YAML changes,
or any other optimization route.

The machine-readable safe summary is
`docs/experiments/evidence/M1_176_FP16_QK_REAL_ACTIVATION_20260904/summary.json`.

## Source and commits

The work started at
`597c991f54aeef6c0aff1107728c37da26a3d26f` on
`exp/M1-132-layered-quality-gate-20260729`. The implementation sequence was:

- `9653e5f`: encode the v2 validation contracts and validator;
- `f5f1994`: preserve the historical M1-137 v1 adapter;
- `4d94c33`: complete TP1 capture derivation and four-rank replay;
- `44a43f3`: bound TP1 long-decode GQA memory;
- `294253b`: replace broad development hashes with lightweight provenance;
- `38466cd`: scope capture validity to the capture-only population;
- `fb6eb11`: reuse a byte-identical runtime overlay;
- `c9cfc82`, `84adb68`, and `12492e9`: add the fixed L3 v2 funnel, complete
  request-population binding, and private-file permissions;
- `e451056`: bind the missing cache-policy identity after the first invalid L3
  control attempt.

Review after the stopped L3 run found that the runner had accepted a diagnostic
checkpoint for a full-model screen. The final harness therefore adds an exact
full-model-path fail-closed check; no experiment was rerun after this finding.

## v1/v2 compatibility and SHA policy

The historical `quality/layered_quality_gate.v1.json`,
`quality/experiment_funnel.v1.json`, and all historical evidence remain
unchanged. `quality/layered_quality_gate.v2.json` and
`quality/experiment_funnel.v2.json` are separate contracts. The validator
dispatches on exact schema and version, rejects unknown combinations, keeps v1
reports on v1 semantics, and does not allow a v1 report to be relabelled v2.

V2 exposes `pass`, `fail`, `inconclusive`, and `invalid` and separates G0
validity, G1 exact semantics, G2 calibrated numerics, G3 distribution review,
G4 paired capability, and G5 paired performance. In particular, distribution
drift is `inconclusive` rather than an operator numeric failure, and the old
universal 1.5x operator threshold and 0.98 top-1 hard gate are absent.

Following the user clarification, SHA-256 is not a general development
freshness requirement. Source uses Git revision and dirty state; runtime uses
install revision, package versions, paths, and byte equality for the two
runtime-critical files. Reports are not individually hashed. This result keeps
SHA-256 only for the three large private activation files and the externally
loaded candidate binary. Private output/token identities used during exact
comparison are stripped from safe evidence.

## Capture and production dispatch boundary

Capture is disabled by default and requires its explicit diagnostic selector.
The production candidate selector is disabled during baseline capture. Two
guards are independent:

- capture-only: TP1, 16 query heads, 2 KV heads, GQA 8:1;
- production fused path: TP4 rank-local, 4 query heads, 1 KV head, GQA 4:1.

Accepting the TP1 diagnostic shape cannot enter the production fused dispatch.
The production guard continues to require the established dtype, device,
layout, block-table, mask, scale, head-dimension 256, block-size 16, and
`17 <= query_length <= 8192` contract. No Q/K/V order, causal position,
softmax scale, token order, or production default was changed.

## TP1-to-TP4 derivation

Model configuration and the real projection layout establish 16 global query
heads, 2 global KV heads, and GQA 8:1. The rank-local mapping is contiguous:

| TP4 rank | Global query heads | Global KV head | Local GQA |
| ---: | --- | ---: | ---: |
| 0 | 0-3 | 0 | 4:1 |
| 1 | 4-7 | 0 | 4:1 |
| 2 | 8-11 | 1 | 4:1 |
| 3 | 12-15 | 1 | 4:1 |

KV is replicated only to the two ranks belonging to its global GQA group.
Physical block IDs are compacted by first occurrence while logical block-table
order is preserved. Unit tests cover contiguous and permuted tables,
partial/full blocks, four-rank reassembly, missing/corrupt manifests, size and
large-file integrity mismatch, shape mismatch, NaN/Inf, and unsupported head
layouts.

## Memory-bounded reference

The old TP1 decode reference could materialize a roughly 2 GiB GQA-broadcast
QK/PV intermediate at 131K. The capture-only fallback now processes each KV
head and a bounded query-head group independently. It never materializes the
full GQA-broadcast attention/PV tensor. Small Torch tests compare it exactly
with the original reference.

The conservative FP32 working bound is 302,841,088 bytes per derived rank and
671,873,024 bytes for the TP1-equivalent reference. It is independent of
context length. The first pre-fix L2 attempt that hit the 2 GiB allocation is
retained as `invalid`; the bounded retry completed without OOM.

## L0 and local validation

Focused tests, every directly affected unit, all available `test_*_unit.py`,
Python syntax, shell syntax, `git diff --check`, and submission preflight were
run. The final broad local unit result before result documentation was 1,552
passed and 20 dependency/environment skips. The foreign untracked M1-164 test
was excluded and none of the five M1-164 files was modified or staged.

Submission preflight passed 10/10. The remote L3 checkout passed 24/24 focused
tests, Python compilation, diff checking, four-card FP16 matmul, TP4 NCCL
all-reduce, and process preflight.

## L1 frozen synthetic result

Only the frozen 16K, 32K, and 64K M1-162 shapes were run, each with query
length 8,176 and five raw trials. No tile, threshold, query-length, or YAML
scan was performed.

| Cell | Baseline median ms | Candidate median ms | Speedup | rel-L2 ratio | max-abs ratio | candidate LSE rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16K | 72.103 | 62.606 | 1.1517x | 1.0000069 | 1.0000000 | 3.66e-8 |
| 32K | 136.101 | 116.755 | 1.1657x | 1.0000068 | 1.0007326 | 3.38e-8 |
| 64K | 264.147 | 225.297 | 1.1724x | 1.0000074 | 1.0000000 | 3.26e-8 |

All outputs and LSE were finite and candidate repeats were bit-exact. L1 is
`pass` under v2 G2. These are operator timings, not end-to-end gains.

Using the historical M1-109 service profile, the conservative inferred
post-M1-109 attention fractions are about 0.276 at 32K and 0.322 at 64K.
Applying the observed FP16-QK speedup gives Amdahl projected gains of about
3.78% and 4.97%. This inference cleared the 2% L3 entry floor but was not
treated as measured service performance.

## L2 real-activation result

Baseline capture ran once on GPU 0 using the four-layer diagnostic checkpoint
derived from the specified model. The captured layer-3 activations are real
model activations, with TP1 shapes Q `[8176,16,256]` and K/V
`[8176,2,256]`. Contexts were 24,576, 57,344, and 122,880 tokens. The three
large private files were 134,063,632, 201,180,688, and 335,414,820 bytes.
They remain under remote `/tmp`, with directory mode 0700 and file mode 0600.

Capture-inclusive wall time was 94.03 seconds. There was no paired no-capture
arm, so pure capture overhead is not identifiable and is deliberately reported
as null.

Four rank replays ran in parallel, reused the same bank, and completed 12/12
G2 cells. Each replay process took about 27.04 seconds; runner wall time was
69.47 seconds. Rank-local speedups ranged from 1.16536x to 1.17118x, with a
1.17068x median. The maximum relative-L2 error ratio was 1.0000123, the maximum
absolute-error ratio was 1.0017094, and candidate LSE relative-L2 ranged from
4.92e-8 to 1.84e-7. Every output was finite and every repeat was exact.

Thus M1-162 on real activations is a v2 G2 `pass`. One baseline-forward timing
cell had 2.30% range dispersion; all other candidate/order-balanced/reference
dispersion maxima were at most 0.19%. Per v2, that single noisy cell does not
override the numerical or aggregate L2 result.

## L3 short TP4 result

The first control attempt is `invalid`: all 72 service and four teacher-forced
HTTP calls returned 200, but report finalization failed because the runtime
manifest omitted the fixed default KV eviction policy. The harness was fixed
once, the attempt was kept separate, and cleanup/postflight passed.

On the fresh retry, control A completed 76/76 requests and all lifecycle gates.
Its control dispatch count was zero. The service population reported 100%
success, 48/48 SLO goodput, TTFT P50/P90/P99 of
0.901/5.942/12.784 seconds, TPOT/ITL P50/P90/P99 of
0.0276/0.0385/0.0411 seconds, and E2E P50/P90/P99 of
1.075/6.158/13.062 seconds. These values are retained only as invalid-run
control diagnostics because post-review found the wrong model path.

Control B then timed out while reading the SSE stream for a partial-prefix
first-sibling request. It completed only 30/72 service requests and 0/4
teacher-forced requests. The candidate was not started. In addition, both
retry arms recorded `/tmp/m1-176-checkpoint.UmD9Hb/model`, not the fixed full
model path. G0 therefore classifies L3 as `invalid` for both identity and
population completeness.

Consequences:

- paired TP4 performance: no conclusion (`invalid`);
- A/A distribution calibration: not completed (`invalid`);
- candidate teacher-forced distribution: not run;
- candidate protocol/cache/capability: no conclusion;
- short TP4 promotion: not authorized.

No control-only latency, operator speedup, or suffix behavior is presented as
a candidate result. No extra paired runs were used.

## Runtime and lifecycle

The BI100 host exposed four Iluvatar BI-V100 GPUs. CoreX was 3.2.3, Torch
2.1.0, vLLM 0.6.3, and Python 3.10.12. The L3 overlay reported Transformers
4.55.3 and install revision `44a43f355cfc0301ffabec4874a6cf29ff2599e3`;
`paged_attn.py` and `qwen3_5.py` were byte-identical to the experiment source.
No runtime-tree hash was used.

Every completed or invalid runner used scoped PID/PGID cleanup and reaping.
After L2 and after both L3 invalid attempts, API/worker/GPU process scans were
empty, fatal scans contained no OOM, CUDA/CoreX error, segfault, worker loss,
or collective reset, and the four GPUs returned to 34,057,748,480 free bytes
each. Final 1024-square FP16 matmul passed on all cards. Final TP4 NCCL
all-reduce returned 10.0 on ranks 0-3 and reaped every child.

Private raw evidence remains only under remote `/tmp`. The local safe exports
are `/tmp/m1-176-fix2-safe-export.ffkhnn`,
`/tmp/m1-176-l3-safe-export.1v0LSc`, and
`/tmp/m1-176-l3-retry-safe-export.PG5Xae`. They contain no raw activation,
prompt, model output, token ID, credential, server log, HMAC key, or extension
binary.

## Final classification

| Dimension | Status | Conclusion |
| --- | --- | --- |
| V2 contract/compatibility | pass | v1 immutable; exact version dispatch; unknown fail-closed |
| L0 validity and harness | pass | local and remote checks passed |
| L1 synthetic numerics | pass | calibrated G2 pass at frozen shapes |
| L2 real-activation numerics | pass | 12/12 rank-local cells passed |
| Kernel timing | pass diagnostic | 15.2%-17.2% operator reduction, not E2E |
| L3 experiment validity | invalid | wrong model identity and incomplete control B population |
| TP4 service performance | invalid | no complete A/A/candidate pair |
| Distribution | invalid | A/A and candidate comparisons not completed |
| Capability | inconclusive | outside this screen and candidate not run |
| Promotion | not authorized | reviewer decision required |

Unfinished work is intentionally limited to a reviewer-approved future L3
rerun with the fixed full model and a complete control/control/candidate
population. This report does not authorize that rerun or any later stage.
