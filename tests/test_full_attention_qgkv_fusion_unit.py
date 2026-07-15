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


def _load_helper(tp_size=2, tp_rank=1):
    tree = ast.parse(QWEN35.read_text(), filename=str(QWEN35))
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_load_full_attention_qgkv_weight"
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
        "get_tensor_model_parallel_world_size": lambda: tp_size,
        "get_tensor_model_parallel_rank": lambda: tp_rank,
    }
    exec(compile(module, str(QWEN35), "exec"), namespace)
    return namespace["_load_full_attention_qgkv_weight"]


@unittest.skipIf(torch is None, f"torch unavailable: {_TORCH_IMPORT_ERROR}")
class FullAttentionQgkvFusionTest(unittest.TestCase):
    def setUp(self):
        self.helper = _load_helper(tp_size=2, tp_rank=1)
        self.config = types.SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=2,
        )
        self.target = "model.layers.3.self_attn.qgkv_proj.weight"
        # Global QG=16 rows, local rank QG=8 rows, K/V=2 rows each.
        self.param = torch.full((12, 3), -1.0)
        self.params = {self.target: self.param}

    def test_qg_rank_slice_and_replicated_kv_load_disjointly(self):
        qg = torch.arange(48).reshape(16, 3).float()
        key = torch.arange(6).reshape(2, 3).float() + 100
        value = key + 100
        for source, weight in (("q_proj", qg), ("k_proj", key),
                               ("v_proj", value)):
            self.assertTrue(self.helper(
                self.params,
                f"model.layers.3.self_attn.{source}.weight",
                weight,
                self.config,
            ))
        self.assertTrue(torch.equal(self.param[:8], qg[8:16]))
        self.assertTrue(torch.equal(self.param[8:10], key))
        self.assertTrue(torch.equal(self.param[10:12], value))

    def test_each_rank_receives_the_column_parallel_qg_slice(self):
        qg = torch.arange(48).reshape(16, 3).float()
        for rank in (0, 1):
            param = torch.zeros(12, 3)
            helper = _load_helper(tp_size=2, tp_rank=rank)
            self.assertTrue(helper(
                {self.target: param},
                "model.layers.3.self_attn.q_proj.weight",
                qg,
                self.config,
            ))
            self.assertTrue(torch.equal(
                param[:8], qg[rank * 8:(rank + 1) * 8]))

    def test_fallback_without_packed_parameter_is_not_handled(self):
        self.assertFalse(self.helper(
            {}, "model.layers.3.self_attn.q_proj.weight",
            torch.zeros(16, 3), self.config))

    def test_invalid_projection_shape_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unexpected.*v_proj"):
            self.helper(
                self.params,
                "model.layers.3.self_attn.v_proj.weight",
                torch.zeros(3, 3),
                self.config,
            )

    def test_nondivisible_qg_fails_closed(self):
        config = types.SimpleNamespace(
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=1,
        )
        helper = _load_helper(tp_size=4, tp_rank=0)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            helper(
                {self.target: torch.zeros(4, 3)},
                "model.layers.3.self_attn.q_proj.weight",
                torch.zeros(6, 3),
                config,
            )

    def test_unrelated_weight_is_not_handled(self):
        self.assertFalse(self.helper(
            self.params, "model.layers.3.mlp.gate.weight",
            torch.zeros(1, 3), self.config))


if __name__ == "__main__":
    unittest.main()
