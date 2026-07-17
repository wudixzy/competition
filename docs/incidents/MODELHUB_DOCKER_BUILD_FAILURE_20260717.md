# ModelHub Docker Build Failure - 2026-07-17

## Status

The competition platform reported that public production commits `c9ed891`
and `7cb514e` failed during image construction. Repository visibility was
public for both submissions, so this incident is distinct from the earlier
private-clone failure. The platform build log is still required to identify
the exact failing command.

Do not attribute the failure to transformers, CoreX compilation, timeout,
disk, or the base-image registry until the final failing log region is
available.

## Evidence from the submitted tree

- Submission preflight on `c9ed891` passes all seven original checks.
- The offline transformers 4.55.3 wheel is 11,269,669 bytes with SHA-256
  `c85e7feace634541e23b3e34d28aa9492d67974b733237ade9eba7c57c0fd1bd`.
- The root Dockerfile still uses the qualified `v1.2.3` BI100 base image and
  invokes `qwen3_6_scripts/patch_ops.sh` with explicit bash.
- Unlike the official reference repository, production `patch_ops.sh` builds
  ten custom CoreX extensions during the Docker layer.
- Rebuilding those exact ten sources on the CoreX 3.2.3 host took 227 seconds
  before pip, patching, Python compilation, image pull, layer export, and
  platform scheduling overhead.

The 227-second compiler path is therefore a material timeout and build
reliability risk, but it is not yet proven to be this platform failure's root
cause.

The second failure on `7cb514e` shows that removing the compiler path was not
by itself sufficient to make the platform build pass. It does not prove that
the prebuilt bundle is invalid: ModelHub `main` points to the expected commit,
the ten binary objects are tracked, and their fixed manifest passes locally.

## Build-hardening candidate

Branch `fix/docker-prebuilt-corex` moves custom extension compilation out of
the evaluator Docker critical path:

1. Rebuild all ten `.so` files from the exact `c9ed891` sources on CoreX
   3.2.3 for `ivcore10`.
2. Store the 2,033,032-byte bundle under
   `qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/`.
3. Verify a fixed SHA-256 manifest before installation.
4. Dynamically load every installed library with CoreX PyTorch during the
   Docker patch layer, failing the image build immediately on ABI mismatch.
5. Keep all CUDA sources and development build scripts in the repository;
   only the submission build path changes.

The freshly built bundle loaded 10/10 libraries on the authoritative CoreX
host. A Docker-equivalent run against a copy of the base image's original
site-package vLLM completed the full `patch_ops.sh` flow in 13 seconds,
including all hashes, dynamic loads, transformers/vLLM patching, import gates,
and Python compilation. The old compiler-only phase took 227 seconds.

Local and remote unit discovery both pass 176 tests with 22 environment skips.
Local and remote submission preflight pass 8/8 including the binary-set and
hash gate. Evidence remains on the CoreX host under
`/tmp/docker-build-fix-validation/`; the production service stayed unchanged
and returned HTTP 200 from both health endpoints after validation.

## No-build-dlopen contingency

Commit `7cb514e` also calls `torch.ops.load_library` for all ten extensions
inside the Docker build layer. This passed on the CoreX host, but the platform
builder may not expose GPU devices or driver libraries. Without the platform
failure tail this is a risk hypothesis, not a confirmed root cause.

Branch `fix/docker-build-no-dlopen` removes that build-host dependency while
retaining the fixed SHA-256 manifest, exact artifact set, non-empty-file gate,
and 64-bit little-endian x86-64 ELF validation. Runtime behavior is unchanged:
the model and paged-attention modules import the `vllm.corex_*` extension
modules during service startup, so an ABI or loader failure still prevents the
service from becoming ready. The branch passes 176 local tests with 22 skips
and submission preflight 8/8. On the CoreX 3.2.3 host, a complete patch against
isolated copies of the base-image vLLM and Transformers packages took 11
seconds, remote preflight passed 8/8, and a separate runtime process loaded all
10 installed extensions. The production service was not restarted and stayed
HTTP 200 on `/health` and `/v1/models` before and after validation.

This qualifies the contingency for a fast-forward into `main`: it removes a
build-only dependency without changing the files installed into vLLM, model
behavior, or the evaluator launch contract. A successful platform build would
support the build-environment hypothesis; another failure still requires the
platform failure tail and must not be attributed to the binaries without that
evidence.

## Remaining gates

The candidate is qualified for merging as a build-reliability fix. To close
the incident's root-cause analysis, still obtain at least one of:

- the platform build log proving the old compiler path, timeout, or related
  resource failure; or
- a clean platform Docker build of this branch from the official `v1.2.3`
  base image.

After merge, rerun the platform image build while the repository is public and
archive the final build log. Runtime TP4 performance is expected to be
unchanged because the candidate installs the same source-built extensions and
does not change `computility-run.yaml` or model code.

For either a successful or failed rerun, retain the complete interval from the
first `[BI100 BUILD]` line through the final Docker error, together with the
failed Docker step number and total build duration. A status-only "image build
failed" result is insufficient to choose safely between registry, file-copy,
dependency, patch-anchor, loader, and platform-timeout failures.
