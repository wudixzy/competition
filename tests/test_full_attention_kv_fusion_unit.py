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
        and node.name == "_load_full_attention_kv_weight"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)

    def default_weight_loader(param, weight):
        if param.size() != weight.size():
            raise AssertionError((param.size(), weight.size()))
        param.copy_(weight)

    namespace = {
        "torch": torch,
        "default_weight_loader": default_weight_loader,
    }
    exec(compile(module, str(QWEN35), "exec"), namespace)
    return namespace["_load_full_attention_kv_weight"]


@unittest.skipIf(torch is None, f"torch unavailable: {_TORCH_IMPORT_ERROR}")
class FullAttentionKvFusionTest(unittest.TestCase):
    def setUp(self):
        self.helper = _load_helper()
        self.config = types.SimpleNamespace(
            num_key_value_heads=2,
            head_dim=3,
        )
        self.target = "model.layers.3.self_attn.kv_proj.weight"
        self.param = torch.full((12, 4), -1.0)
        self.params = {self.target: self.param}

    def test_k_and_v_load_into_disjoint_replicated_slices(self):
        key = torch.arange(24).reshape(6, 4).float()
        value = key + 100
        self.assertTrue(self.helper(
            self.params,
            "model.layers.3.self_attn.k_proj.weight",
            key,
            self.config,
        ))
        self.assertTrue(self.helper(
            self.params,
            "model.layers.3.self_attn.v_proj.weight",
            value,
            self.config,
        ))
        self.assertTrue(torch.equal(self.param[:6], key))
        self.assertTrue(torch.equal(self.param[6:], value))

    def test_fallback_without_merged_parameter_is_not_handled(self):
        self.assertFalse(self.helper(
            {}, "model.layers.3.self_attn.k_proj.weight",
            torch.zeros(6, 4), self.config))

    def test_invalid_projection_shape_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unexpected.*KV"):
            self.helper(
                self.params,
                "model.layers.3.self_attn.v_proj.weight",
                torch.zeros(5, 4),
                self.config,
            )

    def test_unrelated_weight_is_not_handled(self):
        self.assertFalse(self.helper(
            self.params, "model.layers.3.mlp.gate.weight",
            torch.zeros(1, 4), self.config))


if __name__ == "__main__":
    unittest.main()
