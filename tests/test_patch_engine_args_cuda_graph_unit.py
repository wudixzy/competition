import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "qwen3_6_scripts"
PATCH_SCRIPT = SCRIPTS / "patch_engine_args_cuda_graph.py"

VENDOR_BLOCK = """\
class EngineArgs:
    def create_model_config(self):
        return ModelConfig(
            model=self.model,
            enforce_eager=True,
            max_context_len_to_capture=self.max_context_len_to_capture,
        )
"""


def _make_fake_vllm(root: pathlib.Path, arg_utils_text: str) -> pathlib.Path:
    package = root / "vllm"
    engine = package / "engine"
    engine.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (engine / "__init__.py").write_text("")
    target = engine / "arg_utils.py"
    target.write_text(arg_utils_text)
    return target


def _run_patch(
    fake_root: pathlib.Path,
    *args: str,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(fake_root), str(SCRIPTS), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, str(PATCH_SCRIPT), *args],
        cwd=SCRIPTS,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class EngineArgsCudaGraphPatchUnitTest(unittest.TestCase):

    def test_patch_restores_cli_value_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = _make_fake_vllm(root, VENDOR_BLOCK)

            first = _run_patch(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            patched = target.read_text()
            self.assertIn("enforce_eager=self.enforce_eager,", patched)
            self.assertNotIn("enforce_eager=True,", patched)

            second = _run_patch(root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("[skip] already patched", second.stdout)
            self.assertEqual(patched, target.read_text())

            restored = _run_patch(root, "--restore-vendor-eager")
            self.assertEqual(restored.returncode, 0, restored.stderr)
            self.assertEqual(VENDOR_BLOCK, target.read_text())

    def test_patch_fails_fast_for_unknown_vendor_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_fake_vllm(root, "class EngineArgs:\n    pass\n")

            result = _run_patch(root)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("anchor not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
