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
PATCH_SCRIPT = SCRIPTS / "patch_vllm_qwen3_5.py"

_REGISTRY_ANCHOR = (
    '    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),\n'
    '    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),')


def _make_fake_vllm(root: pathlib.Path, registry_text: str) -> pathlib.Path:
    package = root / "vllm"
    models = package / "model_executor" / "models"
    models.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "model_executor" / "__init__.py").write_text("")
    (models / "__init__.py").write_text("")
    (models / "registry.py").write_text(registry_text)
    (models / "qwen3_5.py").write_text(
        "class Qwen3_5ForCausalLM:\n"
        "    ...\n\n"
        "class Qwen3_5MoeForCausalLM:\n"
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
            f"patch_vllm_qwen3_5_unit_{id(fake_root)}", PATCH_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


class RegistryPatchUnitTest(unittest.TestCase):

    def test_registry_alias_patch_installs_qwen36_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            package = _make_fake_vllm(
                root, "MODEL_REGISTRY = {\n" + _REGISTRY_ANCHOR + "\n}\n")
            module = _load_patch_module(root)

            with redirect_stdout(StringIO()):
                module.main()

            registry = (package / "model_executor" / "models" /
                        "registry.py").read_text()
            for name in [
                    "Qwen3ForCausalLM",
                    "Qwen3MoeForCausalLM",
                    "Qwen3_5ForCausalLM",
                    "Qwen3_5MoeForCausalLM",
                    "Qwen3_6ForCausalLM",
                    "Qwen3_6MoeForCausalLM",
            ]:
                self.assertIn(name, registry)
            self.assertIn(
                '"Qwen3_6MoeForCausalLM": ("qwen3_5", '
                '"Qwen3_5MoeForCausalLM")',
                registry,
            )
            self.assertNotIn('("qwen3_moe", "Qwen3MoeForCausalLM")',
                             registry)

            with redirect_stdout(StringIO()):
                module.main()
            self.assertEqual(
                registry,
                (package / "model_executor" / "models" /
                 "registry.py").read_text(),
            )

    def test_registry_alias_patch_fails_fast_when_anchor_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_fake_vllm(root, "MODEL_REGISTRY = {}\n")
            module = _load_patch_module(root)

            with self.assertRaises(RuntimeError):
                with redirect_stdout(StringIO()):
                    module.main()

    def test_registry_verification_does_not_execute_model_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            package = _make_fake_vllm(
                root, "MODEL_REGISTRY = {\n" + _REGISTRY_ANCHOR + "\n}\n")
            marker = root / "model-imported"
            model = package / "model_executor" / "models" / "qwen3_5.py"
            model.write_text(
                f"open({str(marker)!r}, 'w').write('executed')\n\n"
                "class Qwen3_5ForCausalLM:\n"
                "    ...\n\n"
                "class Qwen3_5MoeForCausalLM:\n"
                "    ...\n",
                encoding="utf-8",
            )
            module = _load_patch_module(root)

            with redirect_stdout(StringIO()):
                module.main()

            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
