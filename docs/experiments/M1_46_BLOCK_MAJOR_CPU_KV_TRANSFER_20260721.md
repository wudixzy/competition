# M1-46: Block-major CPU KV transfer

Status: fixed capability and alt1 CoreX transfer gates qualified; the
hash-pinned extension is bundled on the private branch; pressure replay,
model, capacity, dataset, and publication gates pending; no submission
configuration or `main` change.

The private A/B selector is
`BI100_CPU_KV_TRANSFER_LAYOUT=paged|block_major`; it defaults to `paged`,
rejects every other value, and remains forbidden in submission YAML.
The selector is not runnable from a built image until the new extension has
compiled on BI100, passed the data-plane gate, and been added to the hash-pinned
prebuilt bundle. Until then the default `paged` image path remains intact.

## Decision

M1-45 proves that content-addressed CPU KV retention can recover a 65,520-token
prefix after GPU eviction, but the current CoreX 0.6.3 swap path needs 11.584
seconds for that replay. A fixed transfer probe isolates one structural cause:
the production cache is layer-major, so one logical block is split across ten
attention layers and their K/V regions. A CPU-tier promotion invokes the paged
vendor copy for every layer instead of moving one block-major byte range.

The capability experiment compares that production paged path with one
contiguous pinned-CPU/GPU tensor containing exactly the same bytes. It is an
upper-bound data-layout probe, not a replacement kernel and not model-level
qualification.

## Predeclared capability gate

The paired cases must use the same BI100 device name, Torch version, dtype,
shape, byte count, and exactness checks. Both D2H and H2D must be at least
`4.0x` faster at both 65,536 and 131,072 tokens. Any missing field, mismatched
shape, non-finite timing, or inexact probe fails closed.

| Tokens | Bytes/direction | Paged D2H | Contiguous D2H | Speedup | Paged H2D | Contiguous H2D | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 671,088,640 | 472.373 ms | 30.462 ms | 15.507x | 534.119 ms | 24.047 ms | 22.211x |
| 131,072 | 1,342,177,280 | 944.581 ms | 61.304 ms | 15.408x | 1,071.089 ms | 48.038 ms | 22.297x |

All probes were bit-exact and the strict comparator qualified. Evidence:

- `evidence/M1_46_PAGED_TRANSFER_GATE.json`;
- `evidence/M1_46_CONTIGUOUS_TRANSFER_PROBE.json`;
- `evidence/M1_46_TRANSFER_LAYOUT_COMPARISON.json`.

This clears only the decision gate for a block-major transfer implementation.
The measured speedup cannot be used as a model TTFT projection until GPU
pack/scatter cost and scheduler synchronization are included.

## Implementation contract

1. Keep M1-45 content keys, scheduler ownership, GDN/KV intersection, CPU-slot
   admission, and D2H-before-H2D ordering unchanged.
2. Replace the CPU tier's layer-major physical representation with a
   block-major pinned representation. It must consume the same configured
   4-GiB swap budget; a second full CPU copy is forbidden.
3. Add BI100-only GPU pack/scatter operations between the existing layer-major
   GPU cache and a contiguous staging tensor. With the default selector the
   existing exact paged path is unchanged. An explicitly selected
   `block_major` path must fail fast on unsupported dtype, block size, head
   size, layer count, device, or missing extension so an A/B cannot silently
   measure the control implementation.
4. Derive the fixed staging chunk from the immutable command geometry:
   `8192 max_num_batched_tokens / 16 block_size = 512 blocks`. Process larger
   promotions in deterministic 512-block chunks; do not expose a YAML or
   environment tuning threshold.
5. Reuse bounded staging storage and account for it explicitly. It may consume
   reserved device headroom or reduce surplus blocks, but the 262,144-token
   capacity gate must still pass without OOM.
6. Scheduler mappings remain explicit logical source/destination pairs and
   identical on all TP ranks. Worker-local timing may not change eviction or
   admission decisions.
7. The first candidate uses one fixed pack/DMA/scatter pipeline. If it misses
   the gate, exactly one predeclared split-reduction/double-buffer alternative
   may be tested. Failure of both stops this direction without tile or chunk
   scans.

This direction matches the architecture described by vLLM's
[KV offloading documentation](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/),
its [cross-layer KV layout report](https://vllm.ai/blog/2026-01-08-kv-offloading-connector),
and the scheduler-owned transfer plan in
[vLLM RFC #16144](https://github.com/vllm-project/vllm/issues/16144). The
[LMCache systems paper](https://arxiv.org/abs/2510.09665) provides the rationale
for batching and pipelining transfers. M1-46 ports only the required layout and
transfer mechanism to CoreX vLLM 0.6.3; it does not upgrade the runtime.

## Production gates

- Unit: mapping validation, partial final chunk, duplicate/reordered blocks,
  same-slot D2H then H2D, fallback selection, and 512/513-block boundaries.
- Data plane: exact byte parity for 65K and 131K; no NaN/Inf; all four ranks
  receive identical values.
- Transfer: end-to-end pack plus DMA plus scatter is at least `4.0x` faster than
  the existing paged operation in each direction at 65K and 131K.
- Model: the fixed M1-45 pressure replay keeps every response digest equal and
  improves the 11.584-second post-pressure request by at least 20%; immediate
  warm regression remains at most 2%.
- Capacity/stability: fresh-start 235K warm-repeat and 262K capacity pass with
  no OOM, worker loss, collective failure, or loss of 262K support.
- Publication: only the complete 881-request A/B may decide score promotion;
  all original `>=8000`, TTFT, Output TPS, hit-rate, and success gates remain.

Until every applicable gate qualifies, M1-46 stays on a private experiment
branch and remains absent from `computility-run.yaml` and `main`.

## Primary implementation result

Commit `03e453a` compiled once on CoreX 3.2.3. The resulting extension was
191,752 bytes with SHA-256 `e6bbfee496d7...`; it was not added to the prebuilt
bundle because the fixed performance gate failed. All four GPUs passed the
bounded CUDA smoke and the production diagnostic passed full-byte equality,
the 513-block reordered mapping, and installed worker transfer order.

| Tokens | D2H | D2H vs paged | H2D | H2D vs paged |
| ---: | ---: | ---: | ---: | ---: |
| 65,536 | 97.514 ms | 4.844x | 173.628 ms | 3.076x |
| 131,072 | 182.115 ms | 5.187x | 327.411 ms | 3.271x |

D2H passed, but both H2D cases missed the fixed `4.0x` threshold. The primary
candidate is therefore `PERFORMANCE_REJECTED`, despite being bit-exact.
Evidence:

- `evidence/M1_46_BLOCK_MAJOR_PRIMARY.json`;
- `evidence/M1_46_BLOCK_MAJOR_PRIMARY_COMPARISON.json`.

The only permitted alternative is now fixed: two staging slots remove the
per-chunk H2D synchronization, while contiguous CPU-slot runs use one bulk CPU
copy before the same DMA/scatter kernel. The 512-block chunk, kernel launch,
mapping order, numerical gate, and benchmark cases remain unchanged. If this
single alternative misses either direction's `4.0x` threshold, M1-46 stops.

## Fixed alt1 result

Commit `85c325e` compiled once on CoreX 3.2.3. The 196,328-byte extension has
SHA-256
`47c10acfbb3ec7d190c566d73b7616beea1fccc9ac89f336218144211f6fd1a5`.
The 513-block reordered mapping, CPU and GPU byte equality, worker D2H-before-
H2D order, and all fixed timing cases passed.

| Tokens | D2H | D2H vs paged | H2D | H2D vs paged |
| ---: | ---: | ---: | ---: | ---: |
| 65,536 | 72.591 ms | 6.507x | 72.310 ms | 7.387x |
| 131,072 | 166.525 ms | 5.672x | 149.692 ms | 7.155x |

All four directions exceed the predeclared `4.0x` gate, so alt1 is the only
admitted M1-46 data plane. The exact candidate and comparator reports are
`evidence/M1_46_BLOCK_MAJOR_ALT1.json` and
`evidence/M1_46_BLOCK_MAJOR_ALT1_COMPARISON.json`. The tested ELF is included
in the hash-pinned CoreX bundle; Docker still defaults to `paged` because the
selector remains absent from `computility-run.yaml`.

This result unlocks the fixed M1-45 pressure replay and fresh 235K/262K model
gates. It does not by itself prove a model-level TTFT benefit or authorize the
881-request/publication path.
