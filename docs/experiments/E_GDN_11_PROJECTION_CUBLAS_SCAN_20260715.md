# E-GDN-11: GDN projection cuBLAS algorithm scan

## Scope

E-GDN-08 identified the merged input projection and local output projection as
approximately 26% and 14% of one TP4 rank-local GDN decode layer. E-GDN-11
scans CoreX cuBLAS `GemmEx` algorithms for both fixed batch-one FP16 shapes
before considering a production model-layer replacement.

```text
input:  (1, 2048) x weight (3088, 2048)
output: (1, 1024) x weight (2048, 1024)
modes: Hgemm, default, 0-23, 99-115
remote result: /root/competition/E_GDN_11/results/gpu1.json
extension sha256: 13c141d64e563138f3c14a6b5f4ddbb2382bbc88377356f52d860cc6e7a5cf1e
```

Each mode used 20 warmups, 300 iterations per trial, and seven trials on
physical GPU1. The best exact mode also passed 100 random inputs bit-for-bit.

## Result

| Projection | `F.linear` (ms) | Best exact mode | Candidate (ms) | Speedup | Random exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Merged input | 0.133517 | 21 | 0.133069 | 1.0034x | 100/100 |
| Local output | 0.072340 | 20 | 0.071729 | 1.0085x | 100/100 |

No faster nonexact mode exists. Hgemm (`-2`) is the only nonexact path and is
also substantially slower: `0.2140x` for input and `0.2914x` for output, with
maximum absolute differences `0.00177002` and `0.00088501`.

## Decision

`REJECT AS PERFORMANCE WINNER`. Both exact gains are below the 5% primitive
gate and would not justify replacing vLLM linear layers or qualifying new TP
collective behavior. Do not integrate a custom projection wrapper. Future
projection work must fuse a larger adjacent boundary or use a genuinely
different exact GEMV implementation, not another cuBLAS algorithm scan.
