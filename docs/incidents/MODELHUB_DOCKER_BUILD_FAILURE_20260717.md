# ModelHub Docker Build Failure - 2026-07-17

## Status

The competition platform reported that public production commit `c9ed891`
failed during image construction. Repository visibility was public for the
submission, so this incident is distinct from the earlier private-clone
failure. The platform build log is still required to identify the exact
failing command.

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
host. Local unit discovery passes 176 tests with 22 environment skips, and
submission preflight passes 8/8 including the binary-set and hash gate.

## Remaining gates

Before replacing production main, obtain at least one of:

- the platform build log proving the old compiler path, timeout, or related
  resource failure; or
- a clean Docker build of this branch from the official `v1.2.3` base image.

After merge, rerun the platform image build while the repository is public and
archive the final build log. Runtime TP4 performance is expected to be
unchanged because the candidate installs the same source-built extensions and
does not change `computility-run.yaml` or model code.
