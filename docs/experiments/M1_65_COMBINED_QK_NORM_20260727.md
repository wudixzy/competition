# M1-65 Exact Combined Q/K Normalization

## Decision

M1-65 passes its fixed component gate and is promoted only to a default-off
model-path candidate. It does not yet authorize a production default, formal
YAML change, `main` merge, or performance claim.

The candidate combines the contiguous raw query and key heads into one native
`_l2norm` invocation, then splits them before the existing exact CoreX q/k map.
It is restricted to the qualified TP4 decode shape:

```text
batch=1, key_heads=4, value_heads=8, head_dim=128, dtype=FP16
```

All unsupported shapes retain the separate q/k normalization path. The
candidate is controlled by `BI100_GDN_COMBINED_QK_NORM`, whose default is
`0`. The attested `strict-reference-combined-qk` profile enables this variable
while keeping the previously rejected packed GDN and direct MoE paths off.

## Component Evidence

The valid run used:

```text
instance: ssh-73ca29ba
physical GPU: 1
source: b9278e2ce98fd0dd66fae5da56d383caa6fd9559
run: /tmp/m1-65-combined-qk-b9278e2-20260727T104406
```

| Gate | Result |
| --- | ---: |
| Fixed seeds | 2/2 bit-exact |
| Multi-scale sequence | 500/500 bit-exact |
| NaN/Inf | none |
| Relative L2 | 0 |
| Reference median | 0.110109 ms/layer |
| Candidate median | 0.058552 ms/layer |
| Component speedup | 1.8805x |
| Saving | 0.051556 ms/layer |
| Projected 30-layer saving | 1.54669 ms/token |
| Pre/post GPU memory drop | 0 bytes |
| Residual API/worker/GPU processes | none |
| Fatal/timeout scan | pass |

The component qualifier records `component_qualified=true` and
`production_promotion_authorized=false`. Structured evidence is under
[`evidence/M1_65_COMBINED_QK_NORM`](evidence/M1_65_COMBINED_QK_NORM).

Artifact SHA-256 values:

```text
benchmark       bafa537f0c95035650a337d9ac553e25a130d6b2b7e32572f8429594322c813a
qualification   2c21fb3d028a889a2675884a024fe35456bc89fb84179f00165feba56246f739
runtime identity 86c220026e6a59b72d4b25c3cfa1a05823f324e5a9c7203bb82ba400e5a6ef88
service postflight fb5725f032cde20b22d90b9a6db7e1b1def995ee4b683e88ff8ef69cb905bd1e
```

## Production Gate

The next experiment must use one immutable TP4 runtime overlay and compare:

```text
control:   BI100_QUALITY_KERNEL_PROFILE=strict-reference
candidate: BI100_QUALITY_KERNEL_PROFILE=strict-reference-combined-qk
```

Each side must start from a fresh service and pass full functional quality,
cold/warm deterministic output checks, long-context capacity, throughput,
fatal scan, process/GPU postflight, and per-GPU pre/postflight comparison.
Cleanup is process-group scoped: send `SIGTERM`, wait 60 seconds, use
`SIGKILL` only for survivors, and reap children. Any postflight failure makes
the run invalid.

The projected 1.55 ms/token saving is useful but insufficient by itself to
close the current Output TPS or weighted-score gap. End-to-end TP4 evidence is
required before combining M1-65 with another exact structural optimization.

