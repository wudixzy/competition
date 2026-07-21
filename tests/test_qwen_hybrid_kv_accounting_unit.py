from __future__ import annotations

import pathlib
import unittest


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

    def test_default_configs_expose_only_full_attention_caches(self):
        for relative_path, class_name in CONFIGS:
            with self.subTest(config=class_name):
                namespace = _load_config(relative_path)
                config = namespace[class_name]()
                layer_count = config.text_config.num_hidden_layers
                expected = [
                    "attention" if (index + 1) % 4 == 0
                    else "linear_attention"
                    for index in range(layer_count)
                ]
                self.assertEqual(config.layers_block_type, expected)
                self.assertEqual(
                    config.layers_block_type.count("attention"),
                    layer_count // 4,
                )
                self.assertEqual(
                    len(config.text_config.layer_types), layer_count)

    def test_custom_layer_order_preserves_full_attention_ordinals(self):
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
                config = namespace[class_name](text_config=text_config)
                self.assertEqual(config.layers_block_type, expected)

    def test_model_forward_fails_closed_on_surplus_kv_caches(self):
        source = (ROOT / "qwen3_6_scripts/qwen3_5.py").read_text(
            encoding="utf-8")
        self.assertIn("if attn_idx != len(kv_caches):", source)
        self.assertIn("consumed {attn_idx}, received {len(kv_caches)}", source)


if __name__ == "__main__":
    unittest.main()
