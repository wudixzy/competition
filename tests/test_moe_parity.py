import ast
import pathlib
import types
import unittest

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - local env may lack torch
    torch = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


ROOT = pathlib.Path(__file__).resolve().parents[1]
QWEN35 = ROOT / "qwen3_6_scripts" / "qwen3_5.py"


def _reference_experts(hidden_states, router_logits, w13, w2, top_k):
    routing_weights = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(hidden_states.dtype)
    out = torch.zeros_like(hidden_states)
    for token_idx in range(hidden_states.shape[0]):
        token = hidden_states[token_idx:token_idx + 1]
        for pos in range(top_k):
            eid = int(topk_ids[token_idx, pos].item())
            gate_up = F.linear(token, w13[eid])
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
            expert_out = F.linear(act, w2[eid])
            out[token_idx:token_idx + 1] += (
                expert_out * topk_weights[token_idx, pos]).to(out.dtype)
    return out


def _load_production_experts():
    tree = ast.parse(QWEN35.read_text(), filename=str(QWEN35))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5MoeSparseBlock")
    method = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_pure_pytorch_experts")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "F": F,
        "_USE_COREX_MOE_WEIGHT_GATHER": False,
        "_USE_FUSED_MOE_ACTIVATION": False,
        "_USE_COREX_MOE_EXACT_REDUCE": False,
    }
    exec(compile(module, str(QWEN35), "exec"), namespace)
    return namespace["_pure_pytorch_experts"]


@unittest.skipIf(torch is None, f"torch unavailable: {_TORCH_IMPORT_ERROR}")
class MoEParityTest(unittest.TestCase):

    def _assert_shape_matches_reference(self, tokens, hidden, experts, inter,
                                        top_k, seed, router_scale=1.0):
        torch.manual_seed(seed)
        hidden_states = torch.randn(tokens, hidden)
        router_logits = torch.randn(tokens, experts) * router_scale
        w13 = torch.randn(experts, 2 * inter, hidden)
        w2 = torch.randn(experts, hidden, inter)
        fake_self = types.SimpleNamespace(
            top_k=top_k,
            experts=types.SimpleNamespace(w13_weight=w13, w2_weight=w2),
        )

        production = _load_production_experts()
        actual = production(fake_self, hidden_states, router_logits)
        expected = _reference_experts(
            hidden_states, router_logits, w13, w2, top_k)

        self.assertLess(torch.max(torch.abs(actual - expected)).item(), 1e-3)

    def test_pure_pytorch_experts_matches_reference(self):
        self._assert_shape_matches_reference(5, 16, 7, 9, 3, 5678)

    def test_single_token_fast_path_matches_reference(self):
        self._assert_shape_matches_reference(1, 16, 7, 9, 3, 9012)

    def test_sparse_expert_population_matches_reference(self):
        self._assert_shape_matches_reference(17, 16, 31, 8, 2, 3456)

    def test_large_router_logits_match_full_softmax_reference(self):
        self._assert_shape_matches_reference(
            3, 16, 256, 8, 8, 7890, router_scale=80.0)


if __name__ == "__main__":
    unittest.main()
