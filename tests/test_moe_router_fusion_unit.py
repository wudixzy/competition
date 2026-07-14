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


def _load_weight_loader():
    tree = ast.parse(QWEN35.read_text(), filename=str(QWEN35))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Qwen3_5MoeSparseBlock")
    method = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_router_shared_gate_weight_loader")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(QWEN35), "exec"), namespace)
    return namespace["_router_shared_gate_weight_loader"]


@unittest.skipIf(torch is None, f"torch unavailable: {_TORCH_IMPORT_ERROR}")
class MoeRouterFusionTest(unittest.TestCase):
    def setUp(self):
        self.loader = _load_weight_loader()
        self.fake_self = types.SimpleNamespace(num_experts=4)
        self.param = torch.zeros(5, 3)

    def test_router_and_shared_gate_load_disjoint_rows(self):
        router = torch.arange(12).reshape(4, 3)
        shared = torch.arange(3).reshape(1, 3) + 100
        self.loader(self.fake_self, self.param, router, 0)
        self.loader(self.fake_self, self.param, shared, 1)
        self.assertTrue(torch.equal(self.param[:4], router))
        self.assertTrue(torch.equal(self.param[4:], shared))

    def test_invalid_shard_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unexpected.*shard"):
            self.loader(self.fake_self, self.param, torch.zeros(1, 3), 2)

    def test_invalid_shape_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unexpected.*shape"):
            self.loader(self.fake_self, self.param, torch.zeros(3, 3), 0)


if __name__ == "__main__":
    unittest.main()
