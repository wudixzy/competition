import ast
import pathlib
import unittest

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - local environment may lack torch
    torch = None
    F = None
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
        and node.name == "_load_moe_router_shared_gate_weight"
    )
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(QWEN35), "exec"), namespace)
    return namespace["_load_moe_router_shared_gate_weight"]


@unittest.skipIf(torch is None, f"torch unavailable: {_TORCH_IMPORT_ERROR}")
class MoeRouterSharedGateFusionTest(unittest.TestCase):
    def setUp(self):
        self.num_experts = 4
        self.hidden_size = 3
        self.target = "model.layers.0.mlp.gate_and_shared_gate.weight"
        self.param = torch.nn.Parameter(torch.full(
            (self.num_experts + 1, self.hidden_size), -1.0))
        self.params = {self.target: self.param}
        self.helper = _load_helper()

    def test_router_and_shared_gate_fill_disjoint_rows(self):
        router = torch.arange(12).reshape(4, 3).float()
        shared = torch.tensor([[20.0, 21.0, 22.0]])
        self.assertTrue(self.helper(
            self.params,
            "model.layers.0.mlp.gate.weight",
            router,
            self.num_experts,
        ))
        self.assertTrue(self.helper(
            self.params,
            "model.layers.0.mlp.shared_expert_gate.weight",
            shared,
            self.num_experts,
        ))
        self.assertTrue(torch.equal(self.param[:4], router))
        self.assertTrue(torch.equal(self.param[4:], shared))

    def test_fused_projection_is_exact(self):
        torch.manual_seed(20260715)
        hidden = torch.randn(7, self.hidden_size)
        router = torch.randn(self.num_experts, self.hidden_size)
        shared = torch.randn(1, self.hidden_size)
        fused = torch.cat([router, shared], dim=0)
        actual = F.linear(hidden, fused)
        torch.testing.assert_close(
            actual[:, :self.num_experts], F.linear(hidden, router),
            rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            actual[:, self.num_experts:], F.linear(hidden, shared),
            rtol=1e-6, atol=1e-6)

    def test_unrelated_weight_is_not_handled(self):
        self.assertFalse(self.helper(
            self.params,
            "model.layers.0.self_attn.q_proj.weight",
            torch.zeros(1, self.hidden_size),
            self.num_experts,
        ))

    def test_invalid_source_shape_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unexpected MoE gate weight"):
            self.helper(
                self.params,
                "model.layers.0.mlp.gate.weight",
                torch.zeros(self.num_experts - 1, self.hidden_size),
                self.num_experts,
            )

    def test_missing_target_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "missing fused MoE gate"):
            self.helper(
                {},
                "model.layers.0.mlp.shared_expert_gate.weight",
                torch.zeros(1, self.hidden_size),
                self.num_experts,
            )


if __name__ == "__main__":
    unittest.main()
