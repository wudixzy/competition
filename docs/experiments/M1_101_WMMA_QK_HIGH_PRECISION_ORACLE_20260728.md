# M1-101 M1-28 WMMA QK high-precision oracle

Date: 2026-07-28

Status: private single-GPU gate prepared but not yet run. It changes no
production runtime, default, YAML, model, tokenizer, or request semantics.

## Re-audit

M1-28 measured a `1.608223x` QK primitive speedup. All outputs were finite and
the maximum absolute error passed its fixed bound, but the magnitude 1.0 and
2.0 final-output relative L2 values were `1.3834e-5` and `1.8299e-5` against
the vendor FP32 BMM path.

Both QK implementations legally reorder FP32 accumulation. Under policy v2,
vendor BMM is a production control rather than a mathematical oracle. M1-101
therefore permits one unchanged-kernel comparison against a CPU FP64 QK,
softmax, and PV result rounded once to FP16.

## Frozen inputs

The original M1-28 benchmark, build script, and WMMA source are copied
byte-for-byte from
`exp/M1-28-wmma-qk-capability@b03cb39ad23b49eb15728c99b14d9e1a458fb7f5`:

| Artifact | SHA-256 |
|---|---|
| `bench_attention_wmma_qk.py` | `55a4ed735abda6e88f2bbb3f4cc264af1b9629062fb62c9dfc130f683c63895f` |
| `build_corex_attention_wmma_qk_probe.sh` | `9436cd30428f357addf3bcf90d14618a984d48d08f593ac88db70dc6da688958` |
| `corex_attention_wmma_qk_probe.cu` | `08a68ffc068c7f5a21796b32b64e2164c03f7c1b0270e19d862e116abdd3c688` |

The rerun freezes:

- 128 `16 x 32 x 256` QK tiles;
- FP16 Q/K/V, FP32 candidate/control scores, and FP16 final output;
- magnitudes 0.5, 1.0, and 2.0;
- seed `20260718`, with the historical timing offset 100;
- five warmups and twenty paired timing trials;
- alternating control/candidate timing order;
- the original minimum `1.5x` primitive speedup;
- eight fixed CPU threads for the FP64 oracle.

Only the extension path, explicit CoreX device, and private output path are
command-line arguments. There is no tile, magnitude, seed, tolerance, timing,
or launch-geometry scan.

## Numerical gate

For every output row in all 128 tiles, the candidate must be no worse than the
production control against the rounded FP64 oracle for:

- aggregate relative L2, with only the fixed `1e-8` comparison slack;
- maximum row relative L2, with the same slack;
- maximum absolute error, with no slack;
- rounded-oracle mismatch count;
- finite outputs.

Candidate-versus-control score and output differences remain diagnostic. The
report retains scalar metrics only and no raw tensors.

## Decision boundary

A pass authorizes only a separately designed integration-benefit gate. It does
not establish a service gain and cannot authorize production integration,
YAML/default changes, `main` merge, or repository visibility changes.

A failure closes the M1-28 primitive under the revised oracle. No tolerance,
magnitude, seed, tile, layout, dtype, or launch scan follows.
