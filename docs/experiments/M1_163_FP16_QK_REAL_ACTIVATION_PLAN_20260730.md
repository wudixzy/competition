# M1-163 FP16-QK real-activation gate

## Status

The contract, runtime-compatible builder, external-extension binding, and
TP4 shadow runner are frozen before observing M1-163 real-activation results.
GPU0 on `ssh-73ca29ba` remains unhealthy, so execution is pending. No formal
default, YAML, `main`, or production change is authorized.

## Input evidence

M1-162 passed fresh synthetic 16K, 32K, and 64K screens with speedups of
`1.154x`, `1.165x`, and `1.171x`. Candidate error relative to the same FP32
reference was at most `1.000008x` normal FP16 relative-L2 rounding error and
`1.000733x` normal FP16 maximum-absolute rounding error. LSE relative L2 was
at most `3.662e-8`, and repeated candidate outputs were bit-exact.

This authorizes real-activation validation only. It does not establish TP4
service performance or task capability.

## Frozen gate

`quality/fused_prefill_real_activation_adjudication.v2.json` has SHA-256
`ba37338f4d4112a1bd90e3e700334652a66ebb048f3cea7379ed21cdd3f3aceb`.
It retains the M1-138 four-rank sampling matrix and requires:

- finite candidate and FP32 reference values;
- candidate relative-L2 and maximum-absolute error no greater than twice the
  ordinary FP16 rounding error around the same FP32 reference;
- complete, internally consistent evidence on all four ranks;
- malformed, missing, non-finite, or hash-mismatched evidence to fail closed.

Candidate-versus-rounded-reference relative L2 remains recorded but is
diagnostic. Semantic evidence cannot waive a failed numeric envelope.

## Runtime isolation

The candidate is compiled from
`qwen3_6_scripts/corex_fused_paged_prefill_fp16_qk.cu` with the production
loader module name `corex_fused_paged_prefill`. The result is not installed
over the repository prebuilt extension. M1-163 accepts only an explicit file
under `/tmp` that is not group/other writable, computes its SHA-256, and
passes both path and digest to the existing fail-closed external loader.

The old `legacy` and `calibrated` shadow modes are unchanged. The new
`calibrated_v2` report uses a distinct schema and contract version.

## Pending execution

After all four GPUs pass preflight:

```bash
mkdir -m 700 /tmp/m1-163-build-SOURCE
qwen3_6_scripts/build_corex_fused_paged_prefill_fp16_qk_runtime.sh \
  /tmp/m1-163-build-SOURCE

BI100_RUNTIME_SITE_PACKAGES=/path/to/runtime/site-packages \
BI100_RUNTIME_INSTALL_REPORT=/path/to/runtime/install.json \
scripts/run_m1_163_fp16_qk_calibrated_shadow.sh \
  ssh-73ca29ba \
  /tmp/m1-163-build-SOURCE/corex_fused_paged_prefill.so \
  /tmp/m1-163-shadow-SOURCE
```

A pass authorizes only the next short TP4 service A/B. Cold TTFT, cache
transparency, teacher-forced distribution, task capability, 262144 context,
and final performance gates remain required.
