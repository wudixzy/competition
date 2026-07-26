# M1-60 Qwen3.6 Real-Weight Diagnostic Model

## Decision

M1-60 establishes a reusable reduced-depth diagnostic model and TP1/TP2
service flow for Qwen3.6-35B-A3B on BI100.

Two separate decisions apply:

1. `QUALIFIED FOR DIAGNOSTIC USE`: checkpoint conversion, TP1, TP2, API
   surfaces, deterministic prefix recovery, TP2 communication, target-shape
   loader checks, paged KV and CacheEngine all produced valid evidence.
2. `REJECT CURRENT MOE/GDN NUMERICS`: the existing TP4-shape direct MoE and
   packed GDN kernels exceed the new relative L2 limit of `1e-5`. Their speed
   results do not override this rejection.

This result does not authorize a full-model claim, official score, default
switch change, `computility-run.yaml` change, or `main` merge.

Structured evidence is under
[`evidence/M1_60_DIAGNOSTIC`](evidence/M1_60_DIAGNOSTIC). The consolidated
service-pair report is
[`service_pair_qualification.json`](evidence/M1_60_DIAGNOSTIC/service_pair_qualification.json);
the component rejection is
[`qualification.json`](evidence/M1_60_DIAGNOSTIC/components/qualification.json).

## Lineage

Private branch:

```text
exp/M1-60-qwen36-diagnostic-checkpoint-20260727
```

Milestones:

```text
16c261c  exact-shape diagnostic checkpoint builder and verifier
cb929a0  TP1/TP2 service, API, prefix and layer-path gate
99a91d5  TP4 rank-local component numerical/performance gate
eecbf87  production staged-MoE extension capability fix
```

The final runtime tree SHA-256 is:

```text
1791f189c9ab2079e80d7900a94224e099c82bcea5872c881a26d84dc2abf6cf
```

TP1 used `cb929a0` and TP2 used `eecbf87`, but their immutable runtime tree
hashes are identical. The intervening commits add test and qualification
code, not runtime model code.

## Checkpoint

Source:

```text
/root/public-storage/models/Qwen/Qwen3.6-35B-A3B
```

Persistent diagnostic copy:

```text
/root/shared-storage/models/Qwen/Qwen3.6-35B-A3B-diagnostic-4L-real
```

The converter retains one complete `3 linear_attention + 1 full_attention`
cycle and all non-language-layer tensors. It does not deserialize, cast,
quantize or transform retained tensor payloads.

| Contract | Value |
| --- | ---: |
| Language layers | 4 |
| Layer pattern | GDN, GDN, GDN, full attention |
| Hidden size | 2048 |
| Head dim | 256 |
| Query/KV heads | 16 / 2 |
| GDN key/value heads | 16 / 32 |
| Experts / top-k | 256 / 8 |
| Expert/shared intermediate | 512 / 512 |
| Checkpoint dtype | BF16 |
| Max positions | 262144 |
| Visual tensors | 333 |
| MTP tensors | 19 |
| Total tensors | 424 |
| Weight shards | 5 |
| Weight payload | 11,345,363,552 bytes |

Full verification compared every selected payload byte with the source:

```text
qualified=true
full_hash_checked=true
source_payload_bytes_compared=true
tensor_contract_preserved=true
```

The two generated copies have different manifest hashes because their output
paths and generation timestamps differ. Their config, index and all five
weight shard SHA-256 values are identical. The persistent manifest SHA-256 is:

```text
91a3fe34428f253e52467b1c90c088e10e0c02efa9191916ded9cf81ff77c5b1
```

Reproduction:

```bash
python3 scripts/build_qwen36_diagnostic_checkpoint.py \
  --source /root/public-storage/models/Qwen/Qwen3.6-35B-A3B \
  --output /root/shared-storage/models/Qwen/Qwen3.6-35B-A3B-diagnostic-4L-real \
  --cycles 1

python3 scripts/verify_qwen36_diagnostic_checkpoint.py \
  --source /root/public-storage/models/Qwen/Qwen3.6-35B-A3B \
  --checkpoint /root/shared-storage/models/Qwen/Qwen3.6-35B-A3B-diagnostic-4L-real \
  --full-hash --compare-source-bytes \
  --json-out /tmp/qwen36-diagnostic-verify.json
```

The checkpoint files are not committed to Git.

## Service Gates

Instance `ssh-73ca29ba` had three healthy BI100 devices: physical GPU1, GPU2
and GPU3. GPU3 ran TP1; GPU1 and GPU2 ran TP2.

| Gate | TP1 | TP2 |
| --- | ---: | ---: |
| Service qualified | yes | yes |
| TP/NCCL | 1 / n.a. | 2 / pass |
| Shards loaded | 5/5 | 5/5 |
| `max_model_len` | 262144 | 262144 |
| API cases | 7/7 | 7/7 |
| Completed layer traces | 4 | 8 |
| Partial cached tokens | 8176 | 8176 |
| Warm cached tokens | 11600 | 11600 |
| Fatal/OOM/Gloo/worker loss | none | none |
| GPU free bytes before/after | exact | exact |

The seven structural API cases cover:

- model capacity metadata;
- deterministic greedy replay;
- tool message handling;
- reasoning content;
- structured output;
- base64 multimodal input;
- an intentionally invalid empty `messages` request returning `400`.

After timing and path fields are removed, all seven response evidence records
are exactly equal between TP1 and TP2. The prefix primer, partial-cache and
warm-cache responses also share one digest across both TP modes:

```text
3bc63c8024f15fbed6473820347b78895f39fca936a30dd810ab22b9e1ebc98c
```

The four-rank model-input broadcast gate passed, and the installed model
source attests that a requested but absent GDN restore key raises before
state copy or model execution. The live service did not inject a destructive
missing-state fault; that remains a source-level fail-fast attestation rather
than a live scheduler corruption test.

Current reproduction:

```bash
export BI100_RUNTIME_SITE_PACKAGES=/root/m1-60-runtime-eecbf87/site-packages
export SOURCE_MODEL_PATH=/root/public-storage/models/Qwen/Qwen3.6-35B-A3B

PORT=8013 scripts/run_qwen36_diagnostic_gate.sh \
  /root/shared-storage/models/Qwen/Qwen3.6-35B-A3B-diagnostic-4L-real \
  1 3 ssh-73ca29ba /tmp/m1-60-tp1

PORT=8012 scripts/run_qwen36_diagnostic_gate.sh \
  /root/shared-storage/models/Qwen/Qwen3.6-35B-A3B-diagnostic-4L-real \
  2 1,2 ssh-73ca29ba /tmp/m1-60-tp2
```

## Component Gates

The fixed component run used physical GPU3 and synthetic fixed-seed tensors
with the production TP4 rank-local shapes. It did not load the checkpoint.

### Exact Paths

- Packed QGKV loader covered logical TP4 ranks 0, 1, 2 and 3. All q/k/v
  weight slices and outputs were bit-exact.
- Paged KV gather was bit-exact for key, value and attention output at 32K,
  65K, 131K and 235K.
- CacheEngine GPU-to-CPU-to-GPU round trip, same-slot preservation and
  promotion were byte-exact. Invalid mapping and selector inputs failed
  without writes.

Paged KV fixed timings:

| Context | Gather speedup | Gather + attention speedup |
| ---: | ---: | ---: |
| 32K | 2.169x | 1.449x |
| 65K | 2.378x | 1.499x |
| 131K | 3.419x | 1.959x |
| 235K | 6.114x | 2.541x |

These timings establish a component capability only. They do not establish a
235K service TTFT improvement.

### Rejected Paths

The production MoE extension exposes `w13` and `w2_reduce`; it does not expose
the old experimental `w13_silu`. The gate therefore tests the actual staged
production path.

| MoE boundary | Relative L2 | Limit | Result |
| --- | ---: | ---: | --- |
| direct W13 | 7.302e-6 | 1e-5 | pass |
| direct W2 + routed reduce | 3.504e-4 | 1e-5 | reject |
| staged endpoint | 3.511e-4 | 1e-5 | reject |
| staged 500-step aggregate | 3.336e-4 | 1e-5 | reject |

All 500 steps were finite. Fixed and routed speedups were `6.073x` and
`2.990x`, respectively. The speedups do not qualify a numerically rejected
kernel.

| GDN boundary | Result |
| --- | ---: |
| Candidate median | 0.03404 ms |
| Speedup | 5.144x |
| 1000-step output relative L2 | 4.906e-4 |
| Final-state relative L2 | 3.360e-4 |
| Finite steps | 1000/1000 |

GDN also fails the `1e-5` numerical limit. No tolerance was relaxed and no
tile or threshold scan followed.

Reproduction:

```bash
BI100_RUNTIME_SITE_PACKAGES=/root/m1-60-runtime-eecbf87/site-packages \
  scripts/run_qwen36_diagnostic_component_gates.sh \
  3 ssh-73ca29ba /tmp/m1-60-components
```

The expected final return code for the recorded candidate is nonzero because
qualification correctly rejects MoE/GDN numerics. `runner_status.json` records
the qualification stage rather than an infrastructure failure.

## Scope Limits

- Four layers preserve tensor and path contracts but change model capability.
  Generated text is useful for deterministic engineering checks, not quality
  evaluation.
- TP1 and TP2 do not activate the exact TP4 local-shape direct MoE or packed
  GDN dispatch. Those paths were evaluated only by the component probes.
- The service gate confirms a 262144 capacity contract and an 11.6K prefix
  flow. It does not run an end-to-end 235K or near-262K request.
- The checkpoint preserves BF16 source payloads. The existing CoreX/vLLM
  baseline selects FP16 for service execution; M1-60 adds no dtype override.
- No official 881-request workload, model quality benchmark, Output TPS,
  TTFT score or weighted score was evaluated.
- Raw model responses and server logs remain outside Git. Committed evidence
  contains structural summaries, hashes and synthetic component results only.

## Next Step

The diagnostic harness itself is complete and should be reused unchanged.
The next optimization work should target numerical structure, not YAML or tile
scans:

1. Keep direct W13, which passes `1e-5`, and replace or reformulate the W2
   routed reduction so its endpoint and sequence relative L2 pass.
2. Rework packed GDN reduction/update ordering against the reference state
   transition. Do not accept its current speedup without numerical parity.
3. Re-run this fixed component gate once for each revised design.
4. Only after component numerics pass, run the full 40-layer TP4 model,
   262144/235K quality gates and the official-style workload.

The current formal YAML enables the two rejected kernels. Per the experiment
rules, M1-60 does not modify that YAML or `main`; under the new numerical hard
gate those enabled paths are not newly qualified.
