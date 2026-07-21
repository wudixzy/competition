from __future__ import annotations

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


class QwenHybridKvAccountingTest(unittest.TestCase):

    def test_default_configs_preserve_legacy_accounting(self):
        for relative_path, class_name in CONFIGS:
            with self.subTest(config=class_name):
                namespace = _load_config(relative_path)
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
                candidate = namespace["FULL_ATTENTION_KV_ACCOUNTING"]
                with mock.patch.dict(
                        os.environ, {accounting_env: candidate}, clear=False):
                    config = namespace[class_name](text_config=text_config)
                self.assertEqual(config.layers_block_type, expected)

    def test_candidate_default_moe_config_exposes_10_attention_caches(self):
        relative_path, class_name = CONFIGS[1]
        namespace = _load_config(relative_path)
        accounting_env = namespace["HYBRID_KV_ACCOUNTING_ENV"]
        candidate = namespace["FULL_ATTENTION_KV_ACCOUNTING"]
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

    def test_model_forward_fails_closed_on_surplus_kv_caches(self):
        source = (ROOT / "qwen3_6_scripts/qwen3_5.py").read_text(
            encoding="utf-8")
        self.assertIn("if attn_idx != len(kv_caches):", source)
        self.assertIn("consumed {attn_idx}, received {len(kv_caches)}", source)

    def test_private_selector_is_absent_from_submission_yaml(self):
        yaml = (ROOT / "computility-run.yaml").read_text(encoding="utf-8")
        self.assertNotIn("BI100_HYBRID_KV_ACCOUNTING", yaml)


if __name__ == "__main__":
    unittest.main()
