# M1-135 layered historical candidate reassessment

Date: 2026-07-29

Status: evidence review complete. No runtime default, YAML, or `main` change is
authorized by this review.

Superseded interpretation (2026-07-29): the v2 framework in M1-136 narrows the
hard numerical layer to operator/reference comparisons on the same real
activations. Teacher-forced full-model drift is now an equivalence screen and
escalation trigger. The original table remains as an audit record; M1-109 is
reclassified as `distribution-drift-requires-adjudication`, not as a confirmed
operator failure.

## Rule

Historical decisions are classified by the layer that actually stopped them.
A changed greedy suffix alone is trajectory evidence, not an operator or
capability failure. Conversely, task similarity cannot waive a failed numeric,
cache-transparency, protocol, or lifecycle contract. A candidate may also stay
closed when a correct integrated implementation has too little end-to-end
benefit or is dominated by a better implementation.

## Reclassification

| Candidate | Layered finding | Current action |
|---|---|---|
| M1-109 fused softmax | Operator microbenchmark passed and TP4 TTFT gain is material. M1-132 found full-model distribution drift; M1-134 later proved exact A/A repeatability. | Keep open as `distribution-drift-requires-adjudication`. Require real-activation shadow numerics and powered paired capability noninferiority. |
| M1-108 split4 fused prefill | Three TP4 pairs improved cold TTFT by median 3.98% at 65K and 8.64% at 235K, with exact tested outputs and no decode regression. | Retain as the conservative exact-output fallback. It still needs the complete current-generation promotion matrix before any default. |
| M1-47/M1-99 | Operator and dispatcher numerics passed; earlier 20% continuation threshold was too strict. Their useful result was reproduced and superseded by M1-108/M1-109. | Do not rerun unchanged. Preserve as lineage and fallback evidence. |
| M1-91/M1-96 compensated W13 | High-precision-oracle comparison passed and showed the candidate was more accurate than the vendor reduction. | Numerical rejection is rescinded under the corrected oracle. |
| M1-98 integrated compensated W13 | Complete routed boundary improved only 1.102% before model-level dilution. | Remains closed by the end-to-end stop rule, not by output identity or numerical quality. |
| E-PREFIX-08 | Absolute error was small, but relative L2 remained about 3.49e-5 to 3.55e-5, above the frozen operator limit. | Unresolved layer-3 candidate. Permit only the already specified high-precision oracle, after higher-value work. |
| E-ATTN-06 | 100K stress maximum absolute error reached 0.05937. | Hard layer-3 rejection. Semantic evaluation cannot hide this error. |
| E-MOE-04 | Its old 1,000-token hash divergence alone is no longer a sufficient rejection argument. Full routed-path gain was only about 1.05x to 1.09x, and exact E-MOE-10 plus qualified E-MOE-20 supersede it. | Do not rerun unchanged; closed as dominated and low-value, not as a proven capability regression. |
| admission64/direct stale-state paths | Deterministic reuse could skip tokens without a matching recoverable GDN state. | Hard layer-2 rejection. Missing state must fail fast; task scoring cannot waive it. |
| admission64/hybrid64 repair | Current scheduler-owned state identity and aligned fallback passed exact cold/warm and partial-branch checks. | Retain as the cache correctness baseline while performance candidates are evaluated. |

## Priority after M1-134

1. M1-134 A/A passed exactly. Run M1-109 against the same-real-activation
   PyTorch fallback at fixed 65K and 131K production shapes on all four ranks.
2. If that hard numeric layer passes, run predeclared paired capability
   noninferiority; do not choose margins or samples after seeing candidate
   scores. If it fails, semantic scores cannot waive the operator error.
3. Preserve M1-108 as the exact-output fallback while adjudication is open.
4. Revisit E-PREFIX-08 only with its one bounded high-precision oracle. Do not
   spend TP4 capacity on M1-98, E-ATTN-06, or E-MOE-04.

The machine-readable classification is in
`docs/experiments/evidence/M1_135_LAYERED_HISTORICAL_REASSESSMENT_20260729/classification.json`.
