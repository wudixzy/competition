# M1-45: Content-addressed CPU KV tier

Status: implementation in progress; disabled in submission configuration.

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
- TP4 correctness: direct mode retains the existing 131K/256-token equality
  boundary; aligned mode passes 235K/1000-token full equality; 256K capacity is
  unchanged.
- Performance: fixed-order A/B, same image and command, cache cleared between
  groups. Admission requires at least +5 percentage points effective hit and
  +5% weighted proxy, Output TPS P10 at least 20 and no more than 2% relative
  regression. Final publication still requires score at least 8000, TTFT P90
  at most 5 seconds, success at least 99%, and effective hit at least 50%.

Failure of a gate keeps the selector out of YAML and `main`; it does not trigger
capacity, threshold, or launch-parameter scanning.
