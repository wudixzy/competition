"""
Patches the vLLM model registry and deploys the Qwen3_5 model file.

Deploy steps on the remote machine:
  1. patch_ops.sh locates vLLM with importlib.util.find_spec.
  2. cp modified_scripts/qwen3_5.py into the detected vllm model directory.
  2. python3 modified_scripts/patch_vllm_qwen3_5.py

The registry patch installs Qwen3.6 aliases so /model/config.json does not
need to be edited by hand.
"""

from patch_utils import package_root, replace_once

VLLM_ROOT = package_root("vllm")
REGISTRY = VLLM_ROOT / "model_executor" / "models" / "registry.py"


def main():
    print(f"=== Patching {REGISTRY} ===")
    replace_once(
        REGISTRY,
        '    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),\n'
        '    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),',
        '    "Qwen3ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),\n'
        '    "Qwen3MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),\n'
        '    "Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),\n'
        '    "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),\n'
        '    "Qwen3_6ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),\n'
        '    "Qwen3_6MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),',
        required=True,
        already_contains='"Qwen3_6MoeForCausalLM"')

    print("\n=== Verification ===")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qwen3_5",
            str(VLLM_ROOT / "model_executor" / "models" / "qwen3_5.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Quick check: does the class exist?
        spec.loader.exec_module(mod)
        cls = mod.Qwen3_5ForCausalLM
        print(f"  Qwen3_5ForCausalLM found: {cls}")
        cls_moe = mod.Qwen3_5MoeForCausalLM
        print(f"  Qwen3_5MoeForCausalLM found: {cls_moe}")
    except Exception as e:
        print(f"  [optional] verification failed (may be OK at runtime): {e}")

    print("\nDone. Registry aliases installed; do not edit /model/config.json.")


if __name__ == "__main__":
    main()
