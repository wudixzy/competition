import ast
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "qwen3_6_scripts"
PATCH_SCRIPT = SCRIPTS / "patch_corex_swap_blocks.py"

CLEAN_BLOCK = """\
def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    ixf_F.swap_blocks(src, dst, block_mapping)
"""


def make_fake_vllm(root: pathlib.Path, custom_ops: str) -> pathlib.Path:
    package = root / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    path = package / "_custom_ops.py"
    path.write_text(custom_ops, encoding="utf-8")
    return path


def run_patch(fake_root: pathlib.Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(fake_root), str(SCRIPTS), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, str(PATCH_SCRIPT)],
        cwd=SCRIPTS,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class FakeTensor:

    def __init__(self, pairs, *, device="cpu", dtype="int64"):
        self._pairs = pairs
        self.device = types.SimpleNamespace(type=device)
        self.dtype = dtype
        self.shape = (len(pairs), 2)

    def dim(self):
        return 2

    def tolist(self):
        return self._pairs


class FakeTorch:
    Tensor = FakeTensor
    int64 = "int64"


def load_swap_function(source: str, ixformer_functions):
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "swap_blocks")
    namespace = {"torch": FakeTorch, "ixf_F": ixformer_functions}
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 "<patched_custom_ops>", "exec"), namespace)
    return namespace["swap_blocks"]


class CorexSwapBlocksPatchUnitTest(unittest.TestCase):

    def test_patch_is_idempotent_and_fails_on_unknown_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = make_fake_vllm(root, CLEAN_BLOCK)
            first = run_patch(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            patched = target.read_text(encoding="utf-8")
            self.assertIn("vllm_swap_blocks", patched)

            second = run_patch(root)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("[skip] already patched", second.stdout)
            self.assertEqual(target.read_text(encoding="utf-8"), patched)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            make_fake_vllm(root, "def unrelated():\n    pass\n")
            result = run_patch(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("anchor not found", result.stderr)

    def test_legacy_adapter_normalizes_worker_tensor_once_per_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = make_fake_vllm(root, CLEAN_BLOCK)
            result = run_patch(root)
            self.assertEqual(result.returncode, 0, result.stderr)

            calls = []
            legacy = types.SimpleNamespace(
                vllm_swap_blocks=lambda src, dst, mapping: calls.append(
                    (src, dst, mapping)))
            swap_blocks = load_swap_function(
                target.read_text(encoding="utf-8"), legacy)
            mapping = FakeTensor([[3, 7], [4, 9]])
            swap_blocks("src", "dst", mapping)
            self.assertEqual(calls, [("src", "dst", {3: 7, 4: 9})])

    def test_adapter_rejects_ambiguous_or_malformed_mappings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = make_fake_vllm(root, CLEAN_BLOCK)
            result = run_patch(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            legacy = types.SimpleNamespace(vllm_swap_blocks=lambda *args: None)
            swap_blocks = load_swap_function(
                target.read_text(encoding="utf-8"), legacy)

            for mapping in (
                    FakeTensor([[1, 2], [1, 3]]),
                    FakeTensor([[1, 2], [3, 2]]),
                    FakeTensor([[-1, 2]]),
                    FakeTensor([[1, 2]], device="cuda"),
                    FakeTensor([[1, 2]], dtype="int32")):
                with self.subTest(mapping=mapping._pairs):
                    with self.assertRaises(ValueError):
                        swap_blocks("src", "dst", mapping)

    def test_newer_native_symbol_keeps_original_tensor_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            target = make_fake_vllm(root, CLEAN_BLOCK)
            result = run_patch(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = []
            native = types.SimpleNamespace(
                swap_blocks=lambda src, dst, mapping: calls.append(mapping),
                vllm_swap_blocks=lambda *args: self.fail(
                    "legacy symbol must not run when native symbol exists"),
            )
            swap_blocks = load_swap_function(
                target.read_text(encoding="utf-8"), native)
            mapping = object()
            swap_blocks("src", "dst", mapping)
            self.assertEqual(calls, [mapping])


if __name__ == "__main__":
    unittest.main()
