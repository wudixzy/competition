import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "qwen3_6_scripts"
PATCH_SCRIPT = SCRIPTS / "patch_sampler_greedy_fastpath.py"

IMPORT_ANCHOR = """\
import vllm.envs as envs
from vllm.model_executor.sampling_metadata import (SamplingMetadata,
"""

ORIGINAL_BLOCK = """\
        logits = _apply_min_tokens_penalty(logits, sampling_metadata)

        # Apply presence and frequency penalties.
        if do_penalties:
            logits = _apply_penalties(logits, sampling_tensors.prompt_tokens,
                                      sampling_tensors.output_tokens,
                                      sampling_tensors.presence_penalties,
                                      sampling_tensors.frequency_penalties,
                                      sampling_tensors.repetition_penalties)

        # Use float32 to apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits = logits.to(torch.float)
        logits.div_(sampling_tensors.temperatures.unsqueeze(dim=1))

        if do_top_p_top_k and flashinfer_top_k_top_p_sampling is None:
            logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps,
                                        sampling_tensors.top_ks)

        if do_min_p:
            logits = _apply_min_p(logits, sampling_tensors.min_ps)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
"""


def _make_fake_vllm(root: pathlib.Path, sampler_text: str) -> pathlib.Path:
    package = root / "vllm"
    layers = package / "model_executor" / "layers"
    layers.mkdir(parents=True)
    for init in (package / "__init__.py",
                 package / "model_executor" / "__init__.py",
                 layers / "__init__.py"):
        init.write_text("")
    sampler = layers / "sampler.py"
    sampler.write_text(sampler_text)
    return sampler


def _run_patch(fake_root: pathlib.Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(fake_root), str(SCRIPTS), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, str(PATCH_SCRIPT)], cwd=SCRIPTS, env=env,
        text=True, capture_output=True, check=False)


class SamplerGreedyFastPathPatchUnitTest(unittest.TestCase):

    def test_patch_is_idempotent_and_keeps_reference_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = (IMPORT_ANCHOR
                      + "                           SamplingTensors)\n\n"
                      + "class Sampler:\n"
                      + "    def forward(self, logits, sampling_metadata):\n"
                      + ORIGINAL_BLOCK)
            sampler = _make_fake_vllm(root, source)

            first = _run_patch(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            patched = sampler.read_text()
            self.assertIn("use_bi100_greedy_fast_path", patched)
            self.assertIn("BI100_SAMPLER_GREEDY_FASTPATH", patched)
            self.assertIn("probs = torch.softmax", patched)
            self.assertIn("logprobs = torch.log_softmax", patched)
            compile(patched, str(sampler), "exec")

            second = _run_patch(root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("[skip] already patched", second.stdout)
            self.assertEqual(patched, sampler.read_text())

    def test_patch_fails_fast_for_unknown_sampler_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_fake_vllm(root, "class Sampler:\n    pass\n")

            result = _run_patch(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("anchor not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
