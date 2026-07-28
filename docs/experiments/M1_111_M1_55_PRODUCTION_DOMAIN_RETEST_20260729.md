# M1-111 M1-55 production-domain retest

Date: 2026-07-29

Status: harness prepared; BI100 execution is pending completion of the active
M1-109 TP4 service A/B and M1-110 stage profile. This component experiment
cannot authorize a TP4 service candidate, YAML change, `main` merge, or
official-score claim.

## Reason for the retest

M1-55's query-tiled kernel is the repository's existing implementation of the
deep-fusion direction: it reads paged K/V directly, runs FP32 CoreX WMMA QK and
PV, keeps 512-token online-softmax state in shared memory, and avoids
full-query score and split-output workspaces.

The experiment was closed because its fixed small paged case
(`context=240`, `query=16`) had output relative L2 `1.3592e-5`, above the
`1e-5` limit. Production query shapes were never executed. The current fused
prefill selector already falls back for residual warm queries of 8 or 16
tokens, so M1-111 asks a narrower question: does the unchanged best M1-55
implementation qualify when its proposed support domain starts at a
4096-token query?

This is a retest of the exact `a30b6e7` source, not a new tile, split, or
threshold scan. The build changes only the Python extension module name so the
M1-109 baseline and M1-55 candidate can be loaded in one process.

## Fixed matrix

| Case | Context | Query | Total K/V | Gate |
|---|---:|---:|---:|---:|
| Dense | 0 | 8,176 | 8,176 | no more than 2% regression |
| 32K | 24,576 | 8,176 | 32,752 | no more than 2% regression |
| 65K | 65,536 | 8,176 | 73,712 | at least 1.5x |
| 128K | 122,880 | 8,176 | 131,056 | at least 1.5x |
| 235K | 229,376 | 5,616 | 234,992 | at least 1.5x |
| 262K boundary | 253,952 | 8,192 | 262,144 | at least 1.5x |

Every paged case uses a deterministic physical block permutation. All inputs
are FP16 with head dimension 256, block size 16, and GQA 4:1. Execution order
is balanced across the six cells: three run baseline first and three run the
candidate first.

## Gates

The qualified M1-109 fused-softmax binary is the performance and numerical
baseline. Both binaries are compared independently with the existing FP32
reference. Each side uses one warmup and three CUDA-event trials.

Every cell requires:

- finite output and LSE;
- output and LSE relative L2 no greater than `1e-5`;
- output maximum absolute error no greater than `1e-3`;
- the fixed speed gate in the table;
- exact source and binary SHA identities;
- clean four-GPU preflight, postflight, fatal scan, and process-group cleanup.

Small queries remain outside the proposed support domain and must use the
existing fallback in any later integration. Passing this component matrix
would authorize only a guarded runtime integration followed by next-token,
full-output, TP4, functional, and long-context gates.

## Stop rule

If any fixed production shape fails the numerical gate or the unchanged kernel
does not produce the required long-shape speedup, close the unchanged M1-55
route. Do not scan query tiles, PV split counts, numerical tolerances, or YAML
parameters. Use M1-110's measured dominant stage to select a distinct
arithmetic-preserving data flow.
