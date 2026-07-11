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
PATCH_SCRIPT = SCRIPTS / "patch_transformers_qwen3_5.py"

_AUTO_CONFIG_TEXT = (
    "CONFIG_MAPPING_NAMES = [\n"
    '        ("qwen3", "Qwen3Config"),\n'
    "]\n\n"
    "MODEL_NAMES_MAPPING = [\n"
    '        ("qwen3", "Qwen3"),\n'
    "]\n")

_MODELS_INIT_TEXT = (
    "try:\n"
    "    from .qwen3 import *\n"
    "except OptionalDependencyNotAvailable:\n"
    "    pass\n")

_QWEN35_CONFIG = (
    "from transformers.configuration_utils import PretrainedConfig\n\n"
    "class Qwen3_5Config(PretrainedConfig):\n"
    '    model_type = "qwen3_5"\n')

_QWEN35_MOE_CONFIG = (
    "from transformers.configuration_utils import PretrainedConfig\n\n"
    "class _TextConfig:\n"
    "    num_experts = 128\n"
    "    num_experts_per_tok = 8\n"
    "    shared_expert_intermediate_size = 0\n"
    "    num_hidden_layers = 48\n\n"
    "class Qwen3_5MoeConfig(PretrainedConfig):\n"
    '    model_type = "qwen3_5_moe"\n'
    "    def __init__(self, **kwargs):\n"
    "        super().__init__(**kwargs)\n"
    "        self.text_config = _TextConfig()\n")


def _make_fake_transformers(root: pathlib.Path, auto_config_text: str) -> pathlib.Path:
    package = root / "transformers"
    models = package / "models"
    auto = models / "auto"
    qwen35 = models / "qwen3_5"
    qwen35_moe = models / "qwen3_5_moe"
    for path in [package, models, auto, qwen35, qwen35_moe]:
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").write_text("")
    (package / "configuration_utils.py").write_text(
        "class PretrainedConfig:\n"
        "    def __init__(self, **kwargs):\n"
        "        for key, value in kwargs.items():\n"
        "            setattr(self, key, value)\n")
    (auto / "configuration_auto.py").write_text(auto_config_text)
    (models / "__init__.py").write_text(_MODELS_INIT_TEXT)
    (qwen35 / "configuration_qwen3_5.py").write_text(_QWEN35_CONFIG)
    (qwen35_moe / "configuration_qwen3_5_moe.py").write_text(
        _QWEN35_MOE_CONFIG)
    return package


def _clear_import_cache() -> None:
    for name in list(sys.modules):
        if name == "transformers" or name.startswith("transformers."):
            sys.modules.pop(name, None)


def _load_patch_module(fake_root: pathlib.Path):
    _clear_import_cache()
    old_path = list(sys.path)
    sys.path[:0] = [str(fake_root), str(SCRIPTS)]
    importlib.invalidate_caches()
    try:
        spec = importlib.util.spec_from_file_location(
            f"patch_transformers_qwen3_5_unit_{id(fake_root)}",
            PATCH_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


class TransformersPatchUnitTest(unittest.TestCase):

    def test_transformers_patch_registers_qwen35_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            package = _make_fake_transformers(root, _AUTO_CONFIG_TEXT)
            module = _load_patch_module(root)

            with redirect_stdout(StringIO()):
                module.main()

            auto_config = (package / "models" / "auto" /
                           "configuration_auto.py").read_text()
            models_init = (package / "models" / "__init__.py").read_text()
            for token in [
                    '("qwen3_5", "Qwen3_5Config")',
                    '("qwen3_5_moe", "Qwen3_5MoeConfig")',
                    '("qwen3_5", "Qwen3_5")',
                    '("qwen3_5_moe", "Qwen3_5_MoE")',
            ]:
                self.assertIn(token, auto_config)
            self.assertIn("from .qwen3_5 import *", models_init)
            self.assertIn("from .qwen3_5_moe import *", models_init)

            with redirect_stdout(StringIO()):
                module.main()
            self.assertEqual(
                auto_config,
                (package / "models" / "auto" /
                 "configuration_auto.py").read_text(),
            )
            self.assertEqual(
                models_init,
                (package / "models" / "__init__.py").read_text(),
            )

    def test_transformers_patch_fails_fast_when_anchor_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _make_fake_transformers(root, "CONFIG_MAPPING_NAMES = []\n")
            module = _load_patch_module(root)

            with self.assertRaises(RuntimeError):
                with redirect_stdout(StringIO()):
                    module.main()


if __name__ == "__main__":
    unittest.main()
