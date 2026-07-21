#!/bin/bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUNTIME_ROOT=${BI100_BARE_HOST_RUNTIME_ROOT:-$ROOT/bench_runs/m1_49/runtime_overlay}
JSON_OUT=${1:-}
PATCH_STAGE=""
RUNTIME_STAGE=""

BASE_PYTHONPATH=${PYTHONPATH:-}
SYSTEM_PYTHONPATH="/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages"
export PYTHONPATH="$SYSTEM_PYTHONPATH:${BASE_PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"

if [[ $(id -u) -ne 0 ]]; then
    echo "bare-host runtime installation requires root" >&2
    exit 2
fi
if [[ "$RUNTIME_ROOT" != /* ]]; then
    echo "BI100_BARE_HOST_RUNTIME_ROOT must be absolute" >&2
    exit 2
fi
if [[ -e "$RUNTIME_ROOT" || -L "$RUNTIME_ROOT" ]]; then
    echo "runtime destination already exists: $RUNTIME_ROOT" >&2
    exit 2
fi

required=(
    vllm/core/evictor_v2.py
    vllm/core/block/cpu_kv_content_cache.py
    vllm/core/block/cpu_gpu_block_allocator.py
    vllm/core/block/prefix_caching_block.py
    vllm/core/block/block_table.py
    vllm/core/block_manager_v2.py
    qwen3_6_scripts/patch_ops.sh
    qwen3_6_scripts/patch_utils.py
    qwen3_6_scripts/patch_xformers_profile.py
)
for relative in "${required[@]}"; do
    if [[ ! -f "$ROOT/$relative" ]]; then
        echo "required runtime source is missing: $ROOT/$relative" >&2
        exit 2
    fi
done

cleanup() {
    if [[ -n "$PATCH_STAGE" && -d "$PATCH_STAGE" ]]; then
        rm -rf "$PATCH_STAGE"
    fi
    if [[ -n "$RUNTIME_STAGE" && -d "$RUNTIME_STAGE" ]]; then
        rm -rf "$RUNTIME_STAGE"
    fi
}
trap cleanup EXIT

PATCH_STAGE=$(mktemp -d /tmp/bi100-patch-stage.XXXXXX)
cp -a "$ROOT/qwen3_6_scripts/." "$PATCH_STAGE/"
mkdir -p "$PATCH_STAGE/vendor_overrides/vllm/core/block"
install -m 0644 "$ROOT/vllm/core/evictor_v2.py" \
    "$PATCH_STAGE/vendor_overrides/vllm/core/evictor_v2.py"
for name in cpu_kv_content_cache.py cpu_gpu_block_allocator.py \
        prefix_caching_block.py block_table.py; do
    install -m 0644 "$ROOT/vllm/core/block/$name" \
        "$PATCH_STAGE/vendor_overrides/vllm/core/block/$name"
done
install -m 0644 "$ROOT/vllm/core/block_manager_v2.py" \
    "$PATCH_STAGE/vendor_overrides/vllm/core/block_manager_v2.py"

RUNTIME_PARENT=$(dirname "$RUNTIME_ROOT")
RUNTIME_NAME=$(basename "$RUNTIME_ROOT")
mkdir -p "$RUNTIME_PARENT"
RUNTIME_STAGE=$(mktemp -d \
    "$RUNTIME_PARENT/.${RUNTIME_NAME}.staging.XXXXXX")
SITE_PACKAGES="$RUNTIME_STAGE/site-packages"
mkdir -p "$SITE_PACKAGES"

SYSTEM_VLLM_ROOT=$(cd /tmp && python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit("cannot locate system vLLM")
print(Path(spec.origin).resolve().parent)
PY
)
if [[ ! -d "$SYSTEM_VLLM_ROOT" ]]; then
    echo "system vLLM root does not exist: $SYSTEM_VLLM_ROOT" >&2
    exit 2
fi
cp -a "$SYSTEM_VLLM_ROOT" "$SITE_PACKAGES/vllm"

transformers_wheels=(
    "$PATCH_STAGE"/wheels/transformers-4.55.3*.whl
)
if [[ ${#transformers_wheels[@]} -ne 1 \
        || ! -f "${transformers_wheels[0]}" ]]; then
    echo "exactly one offline transformers 4.55.3 wheel is required" >&2
    exit 2
fi
python3 -m pip install --no-index --no-deps \
    --target "$SITE_PACKAGES" "${transformers_wheels[0]}"

# Every patch operation resolves vLLM and Transformers inside the staging
# overlay. System site-packages remain untouched if any later command fails.
export PYTHONPATH="$SITE_PACKAGES:$SYSTEM_PYTHONPATH:${BASE_PYTHONPATH}"
(
    cd "$PATCH_STAGE"
    bash ./patch_ops.sh
)

# Build the expected post-patch block manager from the authoritative source.
# It is intentionally not byte-equal to the base override because patch_ops
# adds the disabled-by-default privacy-safe cache trace afterward.
EXPECTED_ROOT="$PATCH_STAGE/expected_runtime"
mkdir -p "$EXPECTED_ROOT/vllm/core"
touch "$EXPECTED_ROOT/vllm/__init__.py" "$EXPECTED_ROOT/vllm/core/__init__.py"
install -m 0644 "$ROOT/vllm/core/block_manager_v2.py" \
    "$EXPECTED_ROOT/vllm/core/block_manager_v2.py"
PYTHONPATH="$PATCH_STAGE:$EXPECTED_ROOT" \
    python3 "$PATCH_STAGE/patch_block_manager_cache_trace.py"

REPORT_PATH="$RUNTIME_STAGE/install.json"
(
cd /tmp
python3 - "$ROOT" "$SITE_PACKAGES" "$RUNTIME_ROOT" "$REPORT_PATH" \
    "$EXPECTED_ROOT/vllm/core/block_manager_v2.py" <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


root = Path(sys.argv[1]).resolve()
site = Path(sys.argv[2]).resolve()
runtime_root = Path(sys.argv[3])
output = Path(sys.argv[4])
expected_block_manager = Path(sys.argv[5]).resolve()


def package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"cannot locate installed package: {name}")
    return Path(spec.origin).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


vllm_root = package_root("vllm")
transformers_root = package_root("transformers")
if not vllm_root.is_relative_to(site):
    raise SystemExit(f"vLLM resolved outside staged overlay: {vllm_root}")
if not transformers_root.is_relative_to(site):
    raise SystemExit(
        f"Transformers resolved outside staged overlay: {transformers_root}")

checks = {
    "vllm_model": (
        root / "qwen3_6_scripts/qwen3_5.py",
        vllm_root / "model_executor/models/qwen3_5.py",
    ),
    "bi100_profile": (
        root / "qwen3_6_scripts/bi100_profile.py",
        vllm_root / "bi100_profile.py",
    ),
    "paged_attention": (
        root / "qwen3_6_scripts/paged_attn.py",
        vllm_root / "attention/ops/paged_attn.py",
    ),
    "xformers_backend": (
        root / "vllm/attention/backends/xformers.py",
        vllm_root / "attention/backends/xformers.py",
    ),
    "gdn_prefix": (
        root / "qwen3_6_scripts/gdn_prefix.py",
        vllm_root / "gdn_prefix.py",
    ),
    "scheduler": (
        root / "qwen3_6_scripts/scheduler.py",
        vllm_root / "core/scheduler.py",
    ),
    "block_manager": (
        expected_block_manager,
        vllm_root / "core/block_manager_v2.py",
    ),
    "content_cache": (
        root / "vllm/core/block/cpu_kv_content_cache.py",
        vllm_root / "core/block/cpu_kv_content_cache.py",
    ),
    "moe_config": (
        root / "qwen3_6_scripts/qwen3_5_moe/configuration_qwen3_5_moe.py",
        transformers_root / "models/qwen3_5_moe/configuration_qwen3_5_moe.py",
    ),
}
files = {}
qualified = True
for label, (source, installed) in checks.items():
    source_sha = digest(source)
    installed_sha = digest(installed)
    same = source_sha == installed_sha
    qualified = qualified and same
    installed_relative = installed.relative_to(site)
    files[label] = {
        "source_sha256": source_sha,
        "installed_sha256": installed_sha,
        "same": same,
        "installed_path": str(
            runtime_root / "site-packages" / installed_relative),
    }

model_runner = vllm_root / "worker/model_runner.py"
model_runner_text = model_runner.read_text(encoding="utf-8")
profile_patch_present = (
    "self.model_config.get_num_attention_layers(" in model_runner_text
)
qualified = qualified and profile_patch_present
worker = vllm_root / "worker/worker.py"
worker_text = worker.read_text(encoding="utf-8")
startup_profile_guard_patch = (
    "Mark this synthetic pass so BI100_PROFILE can exclude" in worker_text
    and 'os.environ["BI100_IN_STARTUP_PROFILE"] = "1"' in worker_text
)
qualified = qualified and startup_profile_guard_patch
versions = {
    name: importlib.metadata.version(name)
    for name in ("torch", "transformers", "vllm")
}
qualified = qualified and versions["transformers"] == "4.55.3"
report = {
    "schema": "bi100-bare-host-runtime-install-v2",
    "version": 2,
    "qualified": qualified,
    "runtime_root": str(runtime_root),
    "site_packages": str(runtime_root / "site-packages"),
    "system_site_packages_modified": False,
    "versions": versions,
    "files": files,
    "block_manager_base_sha256": digest(
        root / "vllm/core/block_manager_v2.py"),
    "model_runner_sha256": digest(model_runner),
    "worker_sha256": digest(worker),
    "profile_attention_layer_patch": profile_patch_present,
    "startup_profile_guard_patch": startup_profile_guard_patch,
}
rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
output.write_text(rendered, encoding="utf-8")
print(rendered, end="")
raise SystemExit(0 if qualified else 1)
PY
)

# Publishing is a same-filesystem rename. Before this point a failure only
# removes staging directories and cannot leave an active half-patched runtime.
mv "$RUNTIME_STAGE" "$RUNTIME_ROOT"
RUNTIME_STAGE=""

if [[ -n "$JSON_OUT" && "$JSON_OUT" != "$RUNTIME_ROOT/install.json" ]]; then
    mkdir -p "$(dirname "$JSON_OUT")"
    install -m 0644 "$RUNTIME_ROOT/install.json" "$JSON_OUT.tmp"
    mv "$JSON_OUT.tmp" "$JSON_OUT"
fi

echo "[BI100] bare-host runtime overlay published: $RUNTIME_ROOT"
echo "[BI100] export BI100_RUNTIME_SITE_PACKAGES=$RUNTIME_ROOT/site-packages"
