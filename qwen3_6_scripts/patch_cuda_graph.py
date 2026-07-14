"""Restore vLLM's CLI-controlled CUDA Graph selection.

The BI100 base image hardcodes ``enforce_eager=True`` while constructing the
model config, so the fixed evaluator's ``--max-seq-len-to-capture`` argument is
silently ineffective. This patch restores the upstream EngineArgs value. The
E-GRAPH-01 runtime gate must pass before this candidate can be qualified.
"""

from patch_utils import package_root, replace_once


ARG_UTILS = package_root("vllm") / "engine" / "arg_utils.py"


def main() -> None:
    replace_once(
        ARG_UTILS,
        "            enforce_eager=True,\n",
        "            enforce_eager=self.enforce_eager,\n",
        already_contains="            enforce_eager=self.enforce_eager,\n",
    )


if __name__ == "__main__":
    main()
