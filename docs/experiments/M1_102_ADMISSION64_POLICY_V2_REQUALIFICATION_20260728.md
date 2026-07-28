# M1-102 admission64 policy-v2 requalification

Date: 2026-07-28

Status: private performance-screen correction prepared but not yet rerun on the
current source and overlay. It changes no cache implementation, runtime
default, YAML, model, tokenizer, or request semantics.

## Why this candidate reopens

M1-35 compared `admission64/direct` with `fine32/direct` on the frozen
18-request matrix:

| Metric | fine32 | admission64 | Delta |
|---|---:|---:|---:|
| Effective cache hit | 49.9301% | 61.0671% | +11.1370 pp |
| Output TPS P10 | 21.6563 | 21.7783 | +0.56% |
| TTFT P90 | 20.8748 s | 18.0882 s | -13.35% |
| Weighted proxy | 6699.4888 | 6976.7204 | +4.1381% |
| Success | 100% | 100% | 0 pp |

The candidate was closed because the old stage required both at least five
percentage points of hit gain and at least 5% weighted-proxy gain. It missed
the latter by 57.7428 points despite a large hit improvement, lower TTFT, and
non-regressing Output TPS.

M1-96 policy v2 independently corrected that cache screen. Once the candidate
meets the absolute 50% effective-hit floor, repeated proxy evidence may
continue through either:

- effective hit gain of at least two percentage points; or
- weighted proxy gain of at least 3% without reducing effective hit.

Success, Output TPS, TTFT, cache correctness, quality, capacity, and final
official metrics remain separate mandatory gates. Under that predeclared
policy, the historical M1-35 aggregate clears both benefit paths. Historical
evidence authorizes a current-source rerun only; it does not qualify the
current runtime.

## Comparator correction

`scripts/compare_dataset_shaped_policies.py` now reports the two benefit paths
separately and requires their logical OR. The remaining performance screen is
fail-closed on:

- complete, identical 18-request contracts;
- client/server token agreement and one-block target tolerance;
- matching cold/warm salts;
- success at least 99%;
- candidate effective hit at least 50%;
- Output TPS P10 at least 20 and relative regression at most 2%;
- TTFT P90 relative regression at most 2%.

The comparator explicitly leaves `quality_nonregression_qualified`,
`capacity_256k_preserved`, and `final_qualified` unset. A performance-screen
pass cannot substitute for M1-85 full functional/Agent A/B, deterministic
cold/warm cache correctness, long-context capacity, or an official-style
performance run.

## Execution order

1. finish the active M1-99 TP4 run without sharing its GPUs;
2. run the low-cost M1-100 and M1-101 numerical oracles;
3. run M1-85 on the same current source and immutable overlay for complete
   functional and Agent non-regression;
4. run at least three paired, alternating fine32/admission64 current-source
   performance repetitions with scoped process-group cleanup;
5. proceed to official-style validation only if both quality and repeated
   performance evidence pass.

No step authorizes a default, YAML, `main`, or visibility change.
