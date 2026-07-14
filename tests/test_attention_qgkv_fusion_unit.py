import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
QWEN35 = ROOT / "qwen3_6_scripts" / "qwen3_5.py"


def _load_helper():
    tree = ast.parse(QWEN35.read_text(), filename=str(QWEN35))
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_load_attention_qgkv_weight"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": type("Torch", (), {"Tensor": object}),
        "default_weight_loader": lambda *_args: None,
    }
    exec(compile(module, str(QWEN35), "exec"), namespace)
    return namespace["_load_attention_qgkv_weight"]


class FakeParam:
    def __init__(self):
        self.calls = []

        def weight_loader(_param, weight, shard_id):
            self.calls.append((shard_id, weight))

        self.weight_loader = weight_loader


class AttentionQgkvFusionTest(unittest.TestCase):
    def setUp(self):
        self.helper = _load_helper()
        self.target = "model.layers.3.self_attn.qkv_proj.weight"
        self.param = FakeParam()
        self.params = {self.target: self.param}

    def test_original_projection_weights_load_into_qkv_shards(self):
        for source, shard_id in (
                ("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
            weight = object()
            handled = self.helper(
                self.params,
                f"model.layers.3.self_attn.{source}.weight",
                weight,
            )
            self.assertTrue(handled)
            self.assertEqual(self.param.calls[-1], (shard_id, weight))

    def test_unrelated_weight_is_not_handled(self):
        self.assertFalse(self.helper(
            self.params,
            "model.layers.3.mlp.gate.weight",
            object(),
        ))

    def test_vision_qkv_weight_is_not_handled(self):
        self.assertFalse(self.helper(
            self.params,
            "visual.blocks.0.attn.qkv.weight",
            object(),
        ))

    def test_missing_fused_parameter_fails_closed(self):
        with self.assertRaisesRegex(
                ValueError, "missing fused attention projection"):
            self.helper(
                {},
                "model.layers.3.self_attn.q_proj.weight",
                object(),
            )


if __name__ == "__main__":
    unittest.main()
