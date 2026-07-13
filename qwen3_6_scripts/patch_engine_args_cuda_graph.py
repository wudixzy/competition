"""Switch vLLM between CLI-controlled and vendor-forced eager execution.

The CoreX vLLM image hard-codes ``enforce_eager=True`` while constructing
``ModelConfig``.  That silently ignores the evaluator's normal CLI default and
prevents the existing decode-only CUDA Graph path from ever being exercised.
"""

import argparse

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


parser = argparse.ArgumentParser()
parser.add_argument("--restore-vendor-eager", action="store_true")
args = parser.parse_args()

if args.restore_vendor_eager:
    replace_once(
        ARG_UTILS,
        CLI_CONTROLLED_BLOCK,
        VENDOR_BLOCK,
        required=True,
        already_contains="            enforce_eager=True,\n",
    )
else:
    replace_once(
        ARG_UTILS,
        VENDOR_BLOCK,
        CLI_CONTROLLED_BLOCK,
        required=True,
        already_contains="enforce_eager=self.enforce_eager,",
    )
