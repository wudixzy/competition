import ast
import importlib.util
import pathlib
import shutil
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "qwen3_6_scripts" / "patch_sharded_greedy_argmax.py"
LOGITS_SOURCE = ROOT / "vllm" / "model_executor" / "layers" / "logits_processor.py"
SAMPLER_SOURCE = ROOT / "vllm" / "model_executor" / "layers" / "sampler.py"


def load_patch_module():
    scripts = str(SCRIPT.parent)
    import sys
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("patch_sharded_greedy", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ShardedGreedyPatchUnitTest(unittest.TestCase):

    def test_patch_is_syntax_valid_and_idempotent(self):
        module = load_patch_module()
        with tempfile.TemporaryDirectory() as tmp:
            logits = pathlib.Path(tmp) / "logits_processor.py"
            sampler = pathlib.Path(tmp) / "sampler.py"
            shutil.copyfile(LOGITS_SOURCE, logits)
            shutil.copyfile(SAMPLER_SOURCE, sampler)

            module.patch_logits_processor(logits)
            module.patch_sampler(sampler)
            first_logits = logits.read_text()
            first_sampler = sampler.read_text()
            module.patch_logits_processor(logits)
            module.patch_sampler(sampler)

            self.assertEqual(first_logits, logits.read_text())
            self.assertEqual(first_sampler, sampler.read_text())
            ast.parse(first_logits)
            ast.parse(first_sampler)

    def test_gate_and_collective_are_strict(self):
        source = SCRIPT.read_text()
        required = (
            'world_size != 4',
            'sampling_metadata.num_prompts != 0',
            'params.sampling_type != SamplingType.GREEDY',
            'params.min_tokens != 0',
            'params.logprobs is not None',
            'params.logits_processors',
            'tensor_model_parallel_all_reduce(candidates)',
            'nan_mask = torch.isnan(values)',
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
