import ast
import pathlib
import types
import unittest

try:
    import torch
except Exception as exc:  # pragma: no cover - local environment may lack torch
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


ROOT = pathlib.Path(__file__).resolve().parents[1]
QWEN35 = ROOT / "qwen3_6_scripts" / "qwen3_5.py"


def _load_helper():
    tree = ast.parse(QWEN35.read_text(), filename=str(QWEN35))
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_load_gdn_projection_weight"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "default_weight_loader": lambda *_args: None,
    }
    exec(compile(module, str(QWEN35), "exec"), namespace)
    return namespace["_load_gdn_projection_weight"]


class FakeParam:
    def __init__(self):
        self.calls = []

        def weight_loader(_param, weight, shard_id):
            self.calls.append((shard_id, weight.clone()))

        self.weight_loader = weight_loader


@unittest.skipIf(torch is None, f"torch unavailable: {_TORCH_IMPORT_ERROR}")
class GdnProjectionFusionTest(unittest.TestCase):
    def setUp(self):
        self.helper = _load_helper()
        self.config = types.SimpleNamespace(
            linear_num_key_heads=2,
            linear_key_head_dim=3,
            linear_num_value_heads=4,
            linear_value_head_dim=2,
        )
        self.target = "model.layers.0.linear_attn.in_proj_qkvzba.weight"
        self.param = FakeParam()
        self.params = {self.target: self.param}

    def test_qkv_checkpoint_tensor_loads_three_logical_shards(self):
        loaded = torch.arange(20 * 5).reshape(20, 5)
        handled = self.helper(
            self.params,
            "model.layers.0.linear_attn.in_proj_qkv.weight",
            loaded,
            self.config,
        )
        self.assertTrue(handled)
        self.assertEqual([call[0] for call in self.param.calls], [0, 1, 2])
        self.assertTrue(torch.equal(self.param.calls[0][1], loaded[:6]))
        self.assertTrue(torch.equal(self.param.calls[1][1], loaded[6:12]))
        self.assertTrue(torch.equal(self.param.calls[2][1], loaded[12:20]))

    def test_z_b_a_load_into_disjoint_shards(self):
        for source, shard_id, rows in (
                ("in_proj_z", 3, 8),
                ("in_proj_b", 4, 4),
                ("in_proj_a", 5, 4)):
            loaded = torch.arange(rows * 3).reshape(rows, 3)
            handled = self.helper(
                self.params,
                f"model.layers.0.linear_attn.{source}.weight",
                loaded,
                self.config,
            )
            self.assertTrue(handled)
            actual_id, actual_weight = self.param.calls[-1]
            self.assertEqual(actual_id, shard_id)
            self.assertTrue(torch.equal(actual_weight, loaded))

    def test_unrelated_weight_is_not_handled(self):
        self.assertFalse(self.helper(
            self.params,
            "model.layers.0.mlp.gate.weight",
            torch.zeros(1, 1),
            self.config,
        ))

    def test_invalid_qkv_shape_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unexpected fused QKV"):
            self.helper(
                self.params,
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                torch.zeros(19, 5),
                self.config,
            )


if __name__ == "__main__":
    unittest.main()
