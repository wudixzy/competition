# M1-110 fused-prefill stage profile

Date: 2026-07-29

Status: implementation and local static gates are in progress. This experiment
profiles the qualified M1-109 component pipeline; it cannot authorize a TP4
service candidate, YAML change, `main` merge, or official-score claim.

## Question

M1-109 removes the separate split maximum, exponential normalization, split
sum, and online-state-update passes. It still materializes paged K/V tiles and
the complete FP32 score workspace, then invokes separate QK and PV cuBLAS
operations for every 512-token split.

M1-110 measures how the remaining component time is divided among:

- query conversion and workspace initialization;
- paged K/V gather;
- QK;
- causal or tail masking;
- fused split softmax and online-state update;
- PV;
- split-output merge;
- final normalization and conversion.

The result will decide whether the next bounded implementation should fuse
paged gather with QK, eliminate the score workspace with online softmax/PV, or
stop this attention direction and return to the model profile.

## Fixed matrix

The runner assigns one production shape to each of four BI100 GPUs:

| Case | Context | Query |
|---|---:|---:|
| Dense | 0 | 8,176 |
| 65K | 65,536 | 8,176 |
| 128K | 122,880 | 8,176 |
| 235K | 229,376 | 5,616 |

All shapes retain FP16 inputs, head dimension 256, block size 16, GQA 4:1,
and a query length no greater than 8192. The production M1-109 extension and a
separately compiled instrumentation build run on identical seeded inputs.

## Validity gates

The profile qualifies only when:

- the instrumentation build's normal `forward` output and LSE exactly match
  the production binary;
- event-instrumented and uninstrumented calls exactly match;
- output and LSE are finite and each has relative L2 error at most `1e-5`
  against the reference;
- every expected stage boundary is present and stage times sum to the total;
- the uninstrumented profile build is within 5% of the production binary;
- CUDA-event perturbation is at most 15%;
- preflight, postflight, cleanup, and fatal scans all pass.

Host validation and Python overhead are recorded separately and excluded from
the stage shares. Evidence contains tensor shapes, artifact identities,
timings, numerical errors, device health, and return codes. It contains no
prompts, model outputs, request payloads, credentials, or model weights.

## Decision rule

The dominant stage and its absolute time must be consistent across the 65K,
128K, and 235K cases before selecting a deeper-fusion data flow. M1-110 does
not scan tiles or thresholds. A deeper implementation remains restricted to
the fixed production shape and must retain a safe fallback plus independent
component, next-token, full-output, and TP4 service gates.
