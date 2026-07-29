# M1-126 grouped-QK split512 paged prefill

Date: 2026-07-29

Status: implementation and local static validation in progress. CoreX build,
BI100 numerical qualification, and fixed component A/B are pending.

## Decision basis

M1-124 replaced four 512-token QK/PV partitions with one 2048-token QK/PV
partition. Its fixed four-card component A/B was lifecycle-clean but failed
the predeclared numerical and performance gates:

- output relative L2 ranged from `1.133e-5` to `2.267e-5`, above `1e-5`;
- median M1-109/M1-124 speedup was `0.9914x`;
- only the 235K case improved, by approximately `0.22%`.

M1-124 therefore cannot enter a TP4 service experiment.

The M1-113 design allowed one fallback before stopping this route. M1-126
keeps one grouped QK GEMM over at most 2048 key tokens and all four GQA query
heads, but restores M1-109's 512-token FP32 online-softmax and PV boundaries.
This isolates whether reducing four QK calls to one has value without the
floating-point partition change and 2048-token PV behavior that invalidated
M1-124.

## Fixed contract

- FP16 inputs and outputs with FP32 QK, softmax, PV, and accumulation;
- head dimension 256, block size 16, GQA 4:1;
- query length at most 8192 and total sequence length at most 262144;
- one QK call per 2048-token group;
- sequential 512-token online-softmax updates and one PV call per active
  512-token split;
- explicit zero fill and masking for partial tail splits;
- bounded workspaces and no full-sequence logits;
- no runtime selector, request-semantic, model, tokenizer, YAML, or default
  change.

The unchanged fixed component gate compares the qualified M1-109 binary with
M1-126 on dense, 65K, 128K, and 235K shapes. It requires finite outputs,
relative L2 at most `1e-5`, output max-absolute error at most `1e-3`, median
speedup at least `1.10x`, at least three positive cases, no case regression
over 2%, and clean scoped lifecycle gates.

If M1-126 fails numerical or performance qualification, the grouped-QK
route stops. No additional tile, split, or YAML scan is authorized.

Passing this component gate would authorize only a fresh TP4 service A/B.
It would not authorize changing the runtime default, `computility-run.yaml`,
`main`, repository visibility, or making an official-score claim.
