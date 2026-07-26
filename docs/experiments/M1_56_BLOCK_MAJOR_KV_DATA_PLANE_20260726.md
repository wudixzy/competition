# M1-56 Block-Major KV Data Plane

## Conclusion

The fixed double-buffer alternative passed the predeclared three-card component
gate. On physical GPU1, GPU2, and GPU3, both 65K and 131K transfers exceeded
`4.0x` in D2H and H2D relative to the production
`ixformer.functions.vllm_swap_blocks` path. The weakest qualifying result was
GPU1 131K D2H at `4.095x`.

This result qualifies only the isolated KV data plane. It does not establish
model-service throughput, TTFT, cache-hit improvement, 262144 capacity, or
model capability. The extension is not installed by `patch_ops.sh`, is absent
from the prebuilt runtime artifacts, and does not change `main`,
`computility-run.yaml`, or any default behavior.

## Fixed Design

The candidate targets the frozen Qwen3.6 TP4 rank shape:

- 10 full-attention layers;
- K/V planes stored as FP16 `[2, blocks, 4096]` per layer;
- block size 16 and 163,840 bytes per logical block per rank;
- 512-block chunks;
- two fixed 80 MiB GPU/CPU staging pairs;
- a canonical pinned CPU pool in block-major order;
- GPU vector pack/scatter plus CPU row gather/scatter;
- no chunk, tile, buffer-count, or YAML scan.

D2H overlaps the preceding CPU scatter with the next GPU pack and DMA. H2D
overlaps CPU gather with the preceding H2D and GPU scatter. Events guard every
staging-slot reuse. The worker-level D2H-before-H2D ordering remains a separate
production integration gate.

## Correctness Corrections

Review of the single-buffer prototype found that its synthetic values repeated
every 1009 blocks and could hide a wrong block mapping. The qualifying revision
`739adf0` therefore:

1. encodes a unique finite FP16 signature for every logical block;
2. verifies random and permuted source/destination mappings byte-for-byte;
3. checks both `num_blocks` and `-1` mappings fail fast in the GPU kernels;
4. bounds-checks every GPU source and destination ID;
5. records the same-slot check as component call ordering only;
6. checks full-operation timing is consistent with component bounds.

All four cases on all three cards passed D2H/H2D byte exactness, unique
signatures, invalid-map fail-fast, and component same-slot ordering. No fatal,
OOM, Gloo reset, traceback, or worker loss was found.

## Results

| GPU | 65K D2H | 65K H2D | 131K D2H | 131K H2D |
|---|---:|---:|---:|---:|
| GPU1 | 4.440x | 5.335x | 4.095x | 5.388x |
| GPU2 | 4.350x | 5.300x | 4.244x | 5.375x |
| GPU3 | 4.547x | 6.023x | 4.500x | 5.847x |

The original single-buffer path was rejected. Under the same concurrent
three-card load, its 131K D2H speedup was only `2.539x-2.670x`, and several H2D
or 65K D2H cases also missed `4x`. Component timing located the loss in serial
CPU row scatter/gather, which justified exactly one fixed double-buffer
alternative under the stop rule.

Structured summary and raw reports are in
`docs/experiments/evidence/M1_56_BLOCK_MAJOR_KV_DATA_PLANE_20260726.json` and
the three adjacent `M1_56_BLOCK_MAJOR_KV_GPU*_20260726.json` files.

## Production Boundary

The next implementation may be added only behind a non-default internal flag.
It must:

- preserve the current GPU layer-major tensors consumed by attention;
- replace only the CPU tier's physical layout;
- preserve configured CPU block capacity and scheduler block identities;
- complete all D2H work before H2D reuses a GPU slot;
- reject unsupported layer count, dtype, block shape, or mapping;
- account for two 80 MiB GPU and pinned-CPU staging buffers per rank;
- fall back to the existing layer-wise path when the flag is off.

A healthy TP4 host must then pass startup/capacity, swap pressure, identical
cold/warm greedy tokens, same-slot eviction/restoration, 65K/131K/235K and
262144 boundaries, the full functional/quality gate, and end-to-end A/B. Until
then there is no basis to modify the official YAML or claim a score increase.

The architecture is consistent with upstream scheduler-owned swap planning and
cross-layer contiguous transfer work:

- <https://github.com/vllm-project/vllm/issues/16144>
- <https://github.com/vllm-project/vllm/issues/27742>
- <https://vllm.ai/blog/2026-01-08-kv-offloading-connector>
- <https://arxiv.org/abs/2510.09665>

## Validation

- focused M1-56 tests: 10 passed;
- full local suite: 691 passed, 25 skipped;
- submission preflight: 9/9;
- three-card concurrent component gate: 3/3 qualified.
