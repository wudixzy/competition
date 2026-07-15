import ast
import pathlib
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
        and node.name == "_load_full_attention_projection_weight"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "default_weight_loader": lambda *_args: None,
    }
    exec(compile(module, str(QWEN35), "exec"), namespace)
    return namespace["_load_full_attention_projection_weight"]


class FakeParam:
    def __init__(self):
        self.calls = []

        def weight_loader(_param, weight, shard_id):
            self.calls.append((shard_id, weight.clone()))

        self.weight_loader = weight_loader


@unittest.skipIf(torch is None, f"torch unavailable: {_TORCH_IMPORT_ERROR}")
class FullAttentionProjectionFusionTest(unittest.TestCase):
    def setUp(self):
        self.helper = _load_helper()
        self.target = "model.layers.0.self_attn.qgkv_proj.weight"
        self.param = FakeParam()
        self.params = {self.target: self.param}

    def test_checkpoint_projections_load_disjoint_shards(self):
        for source, shard_id in (("q_proj", 0), ("k_proj", 1), ("v_proj", 2)):
            loaded = torch.arange(12).reshape(3, 4) + shard_id * 100
            handled = self.helper(
                self.params,
                f"model.layers.0.self_attn.{source}.weight",
                loaded,
            )
            self.assertTrue(handled)
            actual_id, actual_weight = self.param.calls[-1]
            self.assertEqual(actual_id, shard_id)
            self.assertTrue(torch.equal(actual_weight, loaded))

    def test_fallback_without_merged_parameter_is_not_handled(self):
        handled = self.helper(
            {}, "model.layers.0.self_attn.k_proj.weight", torch.zeros(2, 2))
        self.assertFalse(handled)

    def test_unrelated_weight_is_not_handled(self):
        handled = self.helper(
            self.params, "model.layers.0.mlp.gate.weight", torch.zeros(1, 1))
        self.assertFalse(handled)


if __name__ == "__main__":
    unittest.main()
