from __future__ import annotations

import ast
import os
import pathlib
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = (
    (
        "qwen3_6_scripts/qwen3_5/configuration_qwen3_5.py",
        "Qwen3_5Config",
    ),
    (
        "qwen3_6_scripts/qwen3_5_moe/configuration_qwen3_5_moe.py",
        "Qwen3_5MoeConfig",
    ),
)


class _PreTrainedConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _load_config(relative_path: str) -> dict[str, object]:
    path = ROOT / relative_path
    source = path.read_text(encoding="utf-8")
    import_line = (
        "from ...configuration_utils import "
        "PretrainedConfig as PreTrainedConfig")
    source = source.replace(import_line, "")
    namespace: dict[str, object] = {
        "PreTrainedConfig": _PreTrainedConfig,
        "__name__": f"test_{path.stem}",
    }
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def _load_qwen_cache_validator():
    path = ROOT / "qwen3_6_scripts/qwen3_5.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_qwen_kv_cache_count")
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_validate_qwen_kv_cache_count"]


class QwenHybridKvAccountingTest(unittest.TestCase):

    def test_default_configs_preserve_legacy_accounting(self):
        for relative_path, class_name in CONFIGS:
            with self.subTest(config=class_name):
                namespace = _load_config(relative_path)
                with mock.patch.dict(os.environ, {}, clear=True):
                    config = namespace[class_name]()
                layer_count = config.text_config.num_hidden_layers
                self.assertEqual(
                    config.layers_block_type, ["attention"] * layer_count)
                self.assertEqual(
                    len(config.text_config.layer_types), layer_count)

    def test_candidate_preserves_full_attention_ordinals(self):
        layer_types = [
            "full_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
        ]
        expected = [
            "attention",
            "linear_attention",
            "attention",
            "linear_attention",
        ]
        for relative_path, class_name in CONFIGS:
            with self.subTest(config=class_name):
                namespace = _load_config(relative_path)
                text_class_name = class_name.replace("Config", "TextConfig")
                text_config = namespace[text_class_name](
                    num_hidden_layers=4,
                    layer_types=layer_types,
                )
                accounting_env = namespace["HYBRID_KV_ACCOUNTING_ENV"]
                accounting_config = namespace["HYBRID_KV_ACCOUNTING_CONFIG"]
                candidate = namespace["FULL_ATTENTION_KV_ACCOUNTING"]
                with mock.patch.dict(
                        os.environ, {accounting_env: candidate}, clear=True):
                    config = namespace[class_name](text_config=text_config)
                self.assertEqual(config.layers_block_type, expected)
                self.assertEqual(
                    getattr(config, accounting_config), candidate)

                with mock.patch.dict(os.environ, {}, clear=True):
                    reloaded = namespace[class_name](
                        text_config=text_config,
                        **{
                            accounting_config: candidate,
                            "layers_block_type": expected,
                        },
                    )
                self.assertEqual(reloaded.layers_block_type, expected)
                self.assertEqual(
                    getattr(reloaded, accounting_config), candidate)

                legacy = namespace["LEGACY_KV_ACCOUNTING"]
                with mock.patch.dict(
                        os.environ, {accounting_env: legacy}, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "conflicts"):
                        namespace[class_name](
                            text_config=text_config,
                            **{
                                accounting_config: candidate,
                                "layers_block_type": expected,
                            },
                        )

    def test_candidate_default_moe_config_exposes_10_attention_caches(self):
        relative_path, class_name = CONFIGS[1]
        namespace = _load_config(relative_path)
        accounting_env = namespace["HYBRID_KV_ACCOUNTING_ENV"]
        candidate = namespace["FULL_ATTENTION_KV_ACCOUNTING"]
        with mock.patch.dict(os.environ, {}, clear=True):
            layer_types = namespace[class_name]().text_config.layer_types
        mapped = namespace["_vllm_layers_block_type"](
            layer_types, {accounting_env: candidate})
        self.assertEqual(len(mapped), 40)
        self.assertEqual(mapped.count("attention"), 10)

    def test_invalid_accounting_mode_fails_closed(self):
        for relative_path, _ in CONFIGS:
            namespace = _load_config(relative_path)
            accounting_env = namespace["HYBRID_KV_ACCOUNTING_ENV"]
            with self.assertRaisesRegex(RuntimeError, accounting_env):
                namespace["_vllm_layers_block_type"](
                    ["full_attention"], {accounting_env: "auto"})

    def test_serialized_layers_must_match_serialized_mode(self):
        for relative_path, class_name in CONFIGS:
            namespace = _load_config(relative_path)
            accounting_config = namespace["HYBRID_KV_ACCOUNTING_CONFIG"]
            candidate = namespace["FULL_ATTENTION_KV_ACCOUNTING"]
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                        RuntimeError, "serialized layers_block_type conflicts"):
                    namespace[class_name](**{
                        accounting_config: candidate,
                        "layers_block_type": ["attention"],
                    })

    def test_model_accepts_legacy_and_candidate_cache_counts(self):
        validate = _load_qwen_cache_validator()
        validate(40, [object() for _ in range(40)])
        validate(10, [object() for _ in range(10)])
        with self.assertRaisesRegex(RuntimeError, "configured 10, received 40"):
            validate(10, [object() for _ in range(40)])
        with self.assertRaisesRegex(RuntimeError, "configured 40, received 10"):
            validate(40, [object() for _ in range(10)])

    def test_private_selector_is_absent_from_submission_yaml(self):
        yaml = (ROOT / "computility-run.yaml").read_text(encoding="utf-8")
        self.assertNotIn("BI100_HYBRID_KV_ACCOUNTING", yaml)


if __name__ == "__main__":
    unittest.main()
