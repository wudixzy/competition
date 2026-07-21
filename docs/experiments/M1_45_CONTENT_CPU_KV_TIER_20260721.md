# M1-45: Content-addressed CPU KV tier

Status: allocator metadata and single-GPU data-plane gates qualified; TP4 model
gates pending; disabled in submission configuration.

## Decision

M1-44 established that the production BI100/CoreX paged-KV transfer path is
bit-exact and meets the fixed transfer limits through 131,072 tokens. M1-45
therefore tests a scheduler-owned, inclusive CPU cache for immutable prefix KV
blocks. It does not alter the model, GDN state format, attention kernels, fixed
competition command, or the existing 4 GiB CPU swap allocation.

The experimental selector is `BI100_CPU_KV_OFFLOAD=0|1`. Its default is `0`,
invalid values fail during allocator construction, and it must remain absent
from `computility-run.yaml` until all qualification gates pass.

## Runtime contract

1. The chained 32-byte SHA-256 block digest is the only content identity.
2. The GPU content map is queried before the CPU map.
3. GPU eviction lazily stages `GPU -> CPU` only when the digest has no existing
   CPU copy. A CPU hit stages `CPU -> GPU` while retaining the CPU copy.
4. A CPU slot used as an H2D source or D2H destination is pinned for the whole
   scheduler step. Pending D2H content cannot be read until the next step.
   Once a step claims any H2D source, D2H may consume free CPU slots but cannot
   replace resident CPU content. This preserves later, not-yet-claimed blocks
   of the same sequentially allocated prefix when the CPU tier is saturated.
5. A reused GPU slot may be a D2H source and then an H2D destination in the same
   step. Every worker must execute all D2H transfers before any H2D transfer.
6. Mappings have unique sources and destinations and are produced only by the
   scheduler. Tensor-parallel workers execute identical explicit mappings.
7. Prefix accounting stops at the first non-computed KV block. The effective
   cache hit remains the intersection of this contiguous KV prefix and a live
   scheduler-owned GDN restore key.
8. The content tier and request-level preemption swap cannot share CPU slots.
   With the fixed `max_num_seqs=1` contract, preemption remains recomputation;
   any attempt to use request-level swap while the tier is enabled fails fast.

## Predeclared gates

- Unit lifecycle: deterministic 10,000-step oracle covering insert, inclusive
  hit, LRU replacement, slot pinning, duplicate maps, pending visibility,
  malformed hashes, and environment validation.
- Installation: Docker copies both allocator modules, `patch_ops.sh` installs
  them, and an idempotent worker patch proves D2H precedes H2D.
- Single GPU: cold/warm branching-prefix replay is finite and deterministic;
  the warm result matches the no-offload reference and logs at least one H2D
  promotion and one lazy D2H preservation without timeout or OOM.
- Single-GPU transfer order: one fixed paged-KV case preserves a victim with
  D2H and restores different requested content with H2D through that same GPU
  slot; the victim, requested value, and inclusive CPU source are bit-exact.
- TP4 correctness: direct mode retains the existing 131K/256-token equality
  boundary; aligned mode passes 235K/1000-token full equality; 256K capacity is
  unchanged.
- TP4 eviction pressure: run the same fixed request sequence after separate
  clean server starts. The sequence is a 65,536-token cold target, its immediate
  warm replay, the minimum number of fixed-length unique pressure requests
  required to exceed the log-reported GPU KV capacity, then two target replays.
  The `control` run must report at most one cached block after pressure; the
  `candidate` run must restore all but at most two target blocks. Every target
  response must have the same completion length, finish reason, and message
  digest. Raw prompts and responses are never persisted.
- Performance: fixed-order A/B, same image and command, cache cleared between
  groups. Admission requires at least +5 percentage points effective hit and
  +5% weighted proxy, Output TPS P10 at least 20 and no more than 2% relative
  regression. Final publication still requires score at least 8000, TTFT P90
  at most 5 seconds, success at least 99%, and effective hit at least 50%.

Failure of a gate keeps the selector out of YAML and `main`; it does not trigger
capacity, threshold, or launch-parameter scanning.

## Allocator gate result

Commit `3e2be3f` was installed into the real CoreX vLLM package on the a163
instance. The fixed allocator lifecycle ran once and qualified:

- initial allocation emitted no transfers;
- eviction emitted D2H `(GPU 1 -> CPU 0)`;
- the next content hit emitted H2D `(CPU 0 -> GPU 1)` while preserving its new
  victim with D2H `(GPU 1 -> CPU 1)`;
- both restored block objects and the allocator's contiguous prefix were
  computed `[0, 1]`;
- `BI100_CPU_KV_OFFLOAD=0` returned empty maps and an invalid value failed at
  allocator construction.

Evidence: `evidence/M1_45_CPU_KV_ALLOCATOR_GATE.json`. This gate validates
metadata and mapping ownership only. It does not qualify CoreX transfer order,
model output, performance, TP4 behavior, or the submission selector.

## Same-slot data-plane result

Commit `1582eb9` ran the fixed same-slot operation once on physical GPU 0 after
all four BI-V100 devices passed independent bounded CUDA smoke tests. The test
preserved GPU victim slot 1 into CPU slot 1, then restored different requested
content from CPU slot 0 into that same GPU slot 1. The preserved victim,
restored request, and inclusive CPU source were all bit-exact. Evidence:
`evidence/M1_45_CPU_KV_SAME_SLOT_ORDER.json`.

This closes the ordering and CoreX data-movement gate. It still does not prove
GDN/KV intersection correctness in the full model, model-output equivalence,
TTFT improvement, 881-request gain, or TP4 stability.

## TP4 pressure harness

`tests/cpu_kv_offload_pressure_api.py` implements the predeclared pressure
sequence through the OpenAI-compatible API. It constructs prompts with exact
token counts using the local model tokenizer, persists progress atomically after
every request, and records only token counts, timings, finish reasons, and
SHA-256 response digests. A failed request, non-finite timing, imprecise prompt,
zero-token completion, changed target response, or missed cache threshold makes
the gate fail closed.

The pressure request count is derived once from the startup log's GPU block
count and block size; it is not a tuning parameter. The same count, token
lengths, run identifier, model command, and request order must be used for the
clean-start control and candidate runs. Candidate evidence is admissible only
when the control proves that pressure actually evicted the target from GPU KV.

## TP4 capacity diagnostic

The diagnostic candidate at runtime commit `132718b` reached healthy status on
all four workers with the fixed 262,144-token command. `/health` and
`/v1/models` returned HTTP 200, and a one-token smoke request completed with
`finish_reason=length`. Startup reported 16,878 GPU blocks and 6,553 CPU blocks
at block size 16, corresponding to 270,048 and 104,848 token slots.

Harness commit `f4aa85c` constructed the fixed prompts with the remote local
tokenizer before any pressure run:

- target: exactly 65,536 tokens, 376,731 UTF-8 bytes, 2.082 seconds;
- pressure: exactly 135,040 tokens, 776,331 UTF-8 bytes, 7.589 seconds.

Two pressure prompts occupy 16,880 blocks, exceeding measured GPU capacity by
exactly two blocks. This is the frozen pressure geometry for both clean-start
runs. The diagnostic is not qualification evidence because it was candidate
first, enabled cache tracing, and predates the saturated-promotion protection.
