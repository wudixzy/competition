import importlib
import importlib.util
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "qwen3_6_scripts"
PATCH_SCRIPT = SCRIPTS / "patch_vllm_tool_parser.py"

_INIT_TEXT = (
    "from .mistral_tool_parser import MistralToolParser\n\n"
    "__all__ = [\n"
    '    "MistralToolParser", "Internlm2ToolParser", "Llama3JsonToolParser"\n'
    "]\n")


def _make_fake_vllm(root: pathlib.Path, init_text: str) -> pathlib.Path:
    package = root / "vllm"
    tool_parsers = package / "entrypoints" / "openai" / "tool_parsers"
    tool_parsers.mkdir(parents=True)
    for path in [
            package,
            package / "entrypoints",
            package / "entrypoints" / "openai",
    ]:
        (path / "__init__.py").write_text("")
    (tool_parsers / "__init__.py").write_text(init_text)
    (tool_parsers / "qwen3coder_tool_parser.py").write_text(
        "class Qwen3CoderToolParser:\n"
        "    ...\n")
    return package


def _clear_import_cache() -> None:
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            sys.modules.pop(name, None)


def _load_patch_module(fake_root: pathlib.Path):
    _clear_import_cache()
    old_path = list(sys.path)
    sys.path[:0] = [str(fake_root), str(SCRIPTS)]
    importlib.invalidate_caches()
    try:
        spec = importlib.util.spec_from_file_location(
            f"patch_vllm_tool_parser_unit_{id(fake_root)}", PATCH_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


class ToolParserPatchUnitTest(unittest.TestCase):

    def test_tool_parser_patch_registers_qwen3_coder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            package = _make_fake_vllm(root, _INIT_TEXT)
            module = _load_patch_module(root)

            with redirect_stdout(StringIO()):
                module.main()

            init_file = (package / "entrypoints" / "openai" /
                         "tool_parsers" / "__init__.py")
            patched = init_file.read_text()
            self.assertIn(
                "from .qwen3coder_tool_parser import Qwen3CoderToolParser",
                patched)
            self.assertIn('"Qwen3CoderToolParser"', patched)

            with redirect_stdout(StringIO()):
                module.main()
            self.assertEqual(patched, init_file.read_text())

    def test_tool_parser_patch_fails_fast_when_anchor_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_fake_vllm(root, "__all__ = []\n")
            module = _load_patch_module(root)

            with self.assertRaises(RuntimeError):
                with redirect_stdout(StringIO()):
                    module.main()


if __name__ == "__main__":
    unittest.main()
