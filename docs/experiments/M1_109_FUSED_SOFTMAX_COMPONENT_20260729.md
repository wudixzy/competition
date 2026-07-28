# M1-109 fused split-softmax component A/B

Date: 2026-07-29

Status: the BI100 component gate qualified and authorizes a full-model TP4
service experiment. It does not authorize a YAML change, `main` merge,
official-score claim, or production promotion.

## Candidate

Source commit `354e383efd4199af45e770059bcd415ebf8fcc71` was compiled on
CoreX 3.2.3 for `ivcore10`. The resulting
`corex_fused_paged_prefill.so` is 247,176 bytes with SHA-256
`ad4ea7707bb2f2bfe04e07a7ad5fd58a647232be70a3056937a0d738c8254bff`.
The control is the M1-108 binary with SHA-256
`f654eee2c0677812394ff419d316e7e8c98ed1bcc84853a7f8d2ed5755503009`.

M1-109 replaces the separate split maximum, exponential normalization, split
sum, and online-state update operations with one native normalization pass.
Paged KV gather and the FP32 QK/PV cuBLAS calls remain separate. The candidate
therefore tests a bounded softmax-only data-flow change rather than the later
fully fused paged-attention design.

## Fixed BI100 matrix

`scripts/run_m1_109_fused_softmax_component_ab.sh` assigned one fixed
production shape to each of four BI100 GPUs and alternated old/new execution
order by GPU. Every cell used one warmup and three timed trials, FP16 inputs,
head dimension 256, block size 16, and GQA 4:1. All paged-prefix cells used a
physical block permutation; the zero-prefix dense cell had no pages to
permute. The reference and both extensions used the same seeded tensors.

| Case | Old ms | New ms | Old/new | Output rel-L2 | LSE rel-L2 | Max abs |
|---|---:|---:|---:|---:|---:|---:|
| Dense q8176 | 70.809 | 39.824 | 1.778x | 4.723e-6 | 1.910e-8 | 2.441e-4 |
| 65K q8176 | 572.420 | 293.994 | 1.947x | 6.174e-6 | 1.533e-8 | 1.526e-5 |
| 128K q8176 | 1011.769 | 516.217 | 1.960x | 6.054e-6 | 1.508e-8 | 7.629e-6 |
| 235K q5616 | 1272.078 | 658.461 | 1.932x | 6.625e-6 | 1.663e-8 | 7.629e-6 |

All four cases improved. Median old/new speedup was 1.939x. Outputs and LSE
were finite, maximum output relative L2 was 6.625e-6, maximum LSE relative L2
was 1.910e-8, and maximum absolute output error was 2.441e-4. These pass the
fixed `1e-5` relative-L2 and `1e-3` absolute-error component limits.

## Lifecycle

Four-card compute preflight passed before and after the experiment. Each card
reported 34,057,748,480 free bytes before and after. Final postflight found no
API server, worker, or GPU process; fatal scan was empty. All lifecycle RC
files in the retained evidence are zero.

The evidence directory contains only fixed tensor shapes, timings, numeric
errors, artifact identities, GPU health, and return codes. It contains no
prompts, model outputs, request payloads, credentials, environment dump, or
model weights.

## TP4 service screen

The prebuilt bundle now carries the qualified M1-109 binary, while the private
selector remains disabled by default and absent from `computility-run.yaml`.
The TP4 A/B runner retains three alternating fresh-service pairs and now
covers 32K, 65K, 131K, and 235K.

The service comparator requires:

- exact full-output, first-token, completion-count, and finish-reason identity
  for every cold and warm request, including 235K;
- equal control/candidate warm residuals no greater than 16 prompt tokens,
  proving the fused path cannot run in the warm fallback measurements;
- at least 5% median 235K cold improvement and improvement in at least two
  pairs;
- no more than 2% median cold regression at 32K, 65K, or 131K;
- no more than 0.25 second median or 0.5 second individual warm fallback
  slowdown;
- no more than 2% median or 5% individual Output TPS regression.

Warm relative deltas remain diagnostic because sub-second fallback requests
made the old percentage screen unstable. The absolute limits are an
experiment-continuation screen, not a production or official benchmark gate.

## Decision

Proceed with the fixed full-model TP4 A/B. If it qualifies, run the complete
functional and long-context quality suites before considering any default
change. If the component gain does not produce clear end-to-end benefit,
record that result and stop the softmax-only direction; profile the remaining
paged gather, QK/PV, launch, and synchronization costs before implementing a
deeper fused pipeline.
