# M1-174 query-tiled reassessment plan

## Why reopen this exact candidate

M1-55 implemented the structural target that later profiles continued to
recommend: direct paged KV reads, FP32 WMMA QK and PV, local FP32 online
softmax, and no full-query score or split-output workspace. Its best
single-group PV implementation stopped at a small `context=240, query=16`
cell because candidate-versus-rounded output relative L2 was `1.359e-5`, above
the former fixed `1e-5` gate. Production query lengths were never run.

The layered numeric contract introduced after M1-136 gives that difference a
diagnostic rather than automatically fatal role. It still requires finite
results, FP32-calibrated error no greater than twice ordinary FP16 rounding,
LSE relative L2 at most `1e-5`, and bit-exact candidate repeatability. M1-174
therefore reassesses the exact historical source from
`a30b6e7212286cd613c946b1ca02d8972a198863`; it does not tune the kernel.

## Frozen experiment

The baseline is the qualified M1-162 FP16-QK extension. Physical GPUs 1, 2,
and 3 run the fixed 16K, 32K, and 64K P90 cells with query length 8176 and
fresh fixed seeds. Require all calibrated numerical cells to pass, no cell
below `0.98x`, and median speedup at least `1.08x`.

A pass authorizes only real-activation replay. It does not authorize TP4,
production overlay, YAML, `main`, or an official-score claim. Any compile,
numeric, repeatability, lifecycle, or speed failure closes this exact source
again. No query tile, reduction split, threshold, or YAML scan follows.
