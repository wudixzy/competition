# E-ATTN-05: Context-aware paged gather grid

## Scope

E-ATTN-04 launched one 256-thread block per 256 logical K/V elements, capped
at 65,535 blocks. That exact kernel was already a 100K winner, but 32K-96K
paid unnecessary block scheduling overhead. E-ATTN-05 changes only the launch
grid cap:

```text
seq_len <= 96 * 1024:  256 blocks with a grid-stride loop
seq_len >  96 * 1024:  E-ATTN-04 grid, capped at 65,535 blocks
```

Element indexing, FP16-to-FP32 conversion, output layouts, Python dispatch,
and all subsequent attention arithmetic are unchanged. No evaluator or
environment parameter was added.

## Grid scan

GPU1 scans covered 32,769, 49,152, 65,536, 73,728, 81,920, 90,112, 98,304,
99,500, and 100,000 tokens. A 256-block cap was consistently favorable through
98,304. The original large grid became faster at 99,500 and 100,000, so the
cutover is explicit rather than applying one tuned value to every context.

The final rule was then repeated on GPU1-3. Values below compare cap 256 with
the E-ATTN-04 grid; lower ratios are better.

| GPU | Context | E-ATTN-04 ms | Cap-256 ms | Ratio | Exact |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 65,536 | 7.790 | 7.152 | 0.918 | yes |
| 1 | 90,112 | 9.165 | 8.683 | 0.947 | yes |
| 1 | 98,304 | 10.967 | 10.068 | 0.918 | yes |
| 2 | 65,536 | 7.794 | 7.147 | 0.917 | yes |
| 2 | 90,112 | 9.177 | 8.732 | 0.952 | yes |
| 2 | 98,304 | 10.995 | 10.227 | 0.930 | yes |
| 3 | 65,536 | 7.784 | 7.286 | 0.936 | yes |
| 3 | 90,112 | 9.168 | 8.699 | 0.949 | yes |
| 3 | 98,304 | 10.976 | 10.205 | 0.930 | yes |

At 99,500 and 100,000, cap 256 was slower on every device; the production rule
therefore retains the E-ATTN-04 grid. Every scanned final output was bit-exact.

## Production probe

The production source built successfully with SHA-256:

```text
01d958a7b3664b5a8bb611db93c7df6bcbf3af591ab227464fb754d1ae53b87d
```

Direct `paged_attn.py` on/off probes on GPU1 produced:

| Context | Fallback ms | E-ATTN-05 ms | Speedup | Exact |
| ---: | ---: | ---: | ---: | --- |
| 65,536 | 10.7258 | 7.1363 | 1.5030x | yes |
| 98,304 | 20.3227 | 10.0553 | 2.0211x | yes |
| 99,500 | 19.1004 | 9.4184 | 2.0280x | yes |
| 100,000 | 18.6604 | 9.2576 | 2.0157x | yes |

Across ten full-attention layers, the measured conditional saving is about
35.9 ms/token at 64K, 102.7 ms/token at 96K, and 94.0 ms/token at 100K.
The non-monotonic fallback times reflect the vendor matmul/layout runtime and
are why the decision uses repeated same-length A/B measurements.

Raw cross-device artifacts are not committed:

```text
cross-gpu1.json  2e29d7e8...f6211e2
cross-gpu2.json  869e4a52...0d3ea6a
cross-gpu3.json  a680ae08...ce19706
```

## Decision

`QUALIFY FOR TP4 SERVICE A/B; SUPERSEDES E-ATTN-04`. The change adds 5%-8%
to the already exact 64K-96K candidate, preserves the stronger E-ATTN-04 grid
at 99.5K/100K, and changes no arithmetic. It still does not close the native
decode SIGSEGV incident without a four-card reproduction and soak run.
