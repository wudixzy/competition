"""Restore vLLM's CLI-controlled eager/CUDA Graph selection.

The CoreX vLLM image hard-codes ``enforce_eager=True`` while constructing
``ModelConfig``.  That silently ignores the evaluator's normal CLI default and
prevents the existing decode-only CUDA Graph path from ever being exercised.
"""

from patch_utils import package_root, replace_once


ARG_UTILS = package_root("vllm") / "engine" / "arg_utils.py"

VENDOR_BLOCK = """\
            enforce_eager=True,
            max_context_len_to_capture=self.max_context_len_to_capture,"""

CLI_CONTROLLED_BLOCK = """\
            # Honor the CLI value. The CoreX vendor image hard-coded True here,
            # which disabled vLLM's existing decode-only CUDA Graph path.
            enforce_eager=self.enforce_eager,
            max_context_len_to_capture=self.max_context_len_to_capture,"""


replace_once(
    ARG_UTILS,
    VENDOR_BLOCK,
    CLI_CONTROLLED_BLOCK,
    required=True,
    already_contains="enforce_eager=self.enforce_eager,",
)
