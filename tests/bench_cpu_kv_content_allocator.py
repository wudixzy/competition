#!/usr/bin/env python3
"""Fixed metadata lifecycle gate for the M1-45 CPU KV content tier."""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out")
    args = parser.parse_args()

    os.environ["BI100_CPU_KV_OFFLOAD"] = "1"

    from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
    from vllm.utils import Device

    allocator = CpuGpuBlockAllocator.create(
        allocator_type="prefix_caching",
        num_gpu_blocks=2,
        num_cpu_blocks=2,
        block_size=2,
    )

    allocator.begin_prefix_cache_step()
    first = allocator.allocate_immutable_blocks(
        None, [[1, 2], [3, 4]], Device.GPU)
    allocator.mark_blocks_as_computed([])
    initial_maps = allocator.get_and_reset_prefix_swaps()
    for block in reversed(first):
        allocator.free(block)

    allocator.begin_prefix_cache_step()
    replacement = allocator.allocate_immutable_blocks(
        None, [[9, 10]], Device.GPU)
    preserved_maps = allocator.get_and_reset_prefix_swaps()
    allocator.mark_blocks_as_computed([])
    for block in reversed(replacement):
        allocator.free(block)

    allocator.begin_prefix_cache_step()
    restored = allocator.allocate_immutable_blocks(
        None, [[1, 2], [3, 4]], Device.GPU)
    restored_ids = [block.block_id for block in restored]
    computed_ids = allocator.get_computed_block_ids(
        [], restored_ids, skip_last_block_id=False)
    restored_maps = allocator.get_and_reset_prefix_swaps()

    swap_in, swap_out = restored_maps
    assertions = {
        "initial_maps_empty": initial_maps == ([], []),
        "lazy_d2h_recorded": (
            not preserved_maps[0] and len(preserved_maps[1]) == 1),
        "cpu_hit_h2d_recorded": len(swap_in) == 1,
        "victim_d2h_recorded": len(swap_out) == 1,
        "d2h_precedes_h2d_same_gpu_slot": (
            len(swap_in) == 1 and len(swap_out) == 1
            and swap_out[0][0] == swap_in[0][1]),
        "restored_prefix_computed": computed_ids == restored_ids,
        "restored_blocks_marked_computed": all(
            block.computed for block in restored),
    }
    result = {
        "version": 1,
        "qualified": all(assertions.values()),
        "assertions": assertions,
        "initial_maps": initial_maps,
        "preserved_maps": preserved_maps,
        "restored_maps": restored_maps,
        "restored_ids": restored_ids,
        "computed_ids": computed_ids,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as output:
            output.write(payload + "\n")
    if not result["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
