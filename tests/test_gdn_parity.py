import ast
import pathlib
import types
import typing
import unittest
from functools import lru_cache

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


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _reference_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    seq_len = key.shape[2]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    scale = 1.0 / (query.shape[-1]**0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(
        mask_upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    batch, num_heads, _, _, k_dim = key.shape
    v_dim = value.shape[-1]
    total_len = seq_len + pad
    last_state = (
        torch.zeros(batch, num_heads, k_dim, v_dim, dtype=value.dtype,
                    device=value.device)
        if initial_state is None else initial_state.to(value))
    core_out = torch.zeros_like(value)
    mask_upper2 = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1)

    for i in range(total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) *
                  decay_mask[:, :, i]).masked_fill_(mask_upper2, 0)
        v_prime = k_cumdecay[:, :, i] @ last_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_state
        core_out[:, :, i] = attn_inter + attn_i @ v_new
        last_state = (
            last_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None])
            .transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_state = None
    core_out = core_out.reshape(batch, num_heads, -1, v_dim)[:, :, :seq_len]
    core_out = core_out.transpose(1, 2).contiguous()
    return core_out, last_state


def _load_production_chunk_rule():
    tree = ast.parse(QWEN35.read_text(), filename=str(QWEN35))
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_l2norm",
            "_gdn_solve_identity",
            "_solve_gdn_lower_triangular",
            "_torch_chunk_gated_delta_rule",
        ):
            wanted.append(node)
    module = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "F": F,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
        "lru_cache": lru_cache,
        "ixformer_functions": types.SimpleNamespace(solve=torch.linalg.solve),
    }
    exec(compile(module, str(QWEN35), "exec"), namespace)
    return namespace["_torch_chunk_gated_delta_rule"]


@unittest.skipIf(torch is None, f"torch unavailable: {_TORCH_IMPORT_ERROR}")
class GatedDeltaNetParityTest(unittest.TestCase):

    def test_chunk_gated_delta_rule_matches_reference(self):
        torch.manual_seed(1234)
        batch, seq, heads, k_dim, v_dim = 2, 17, 3, 8, 8
        query = torch.randn(batch, seq, heads, k_dim)
        key = torch.randn(batch, seq, heads, k_dim)
        value = torch.randn(batch, seq, heads, v_dim)
        g = -torch.rand(batch, seq, heads)
        beta = torch.rand(batch, seq, heads)
        initial_state = torch.randn(batch, heads, k_dim, v_dim)

        production = _load_production_chunk_rule()
        actual, actual_state = production(
            query, key, value, g, beta, chunk_size=8,
            initial_state=initial_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True)
        expected, expected_state = _reference_chunk_gated_delta_rule(
            query, key, value, g, beta, chunk_size=8,
            initial_state=initial_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True)

        self.assertLess(torch.max(torch.abs(actual - expected)).item(), 1e-3)
        self.assertLess(
            torch.max(torch.abs(actual_state - expected_state)).item(), 1e-3)

    def test_chained_boundary_split_matches_unsplit_rule(self):
        torch.manual_seed(20260712)
        batch, seq, heads, dim, split = 1, 79, 2, 16, 63
        query = torch.randn(batch, seq, heads, dim)
        key = torch.randn(batch, seq, heads, dim)
        value = torch.randn(batch, seq, heads, dim)
        g = -torch.rand(batch, seq, heads)
        beta = torch.rand(batch, seq, heads)
        initial_state = torch.randn(batch, heads, dim, dim)
        production = _load_production_chunk_rule()

        full_out, full_state = production(
            query, key, value, g, beta, chunk_size=64,
            initial_state=initial_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True)
        first_out, boundary_state = production(
            query[:, :split], key[:, :split], value[:, :split],
            g[:, :split], beta[:, :split], chunk_size=64,
            initial_state=initial_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True)
        second_out, split_state = production(
            query[:, split:], key[:, split:], value[:, split:],
            g[:, split:], beta[:, split:], chunk_size=64,
            initial_state=boundary_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True)
        split_out = torch.cat([first_out, second_out], dim=1)

        self.assertLess(torch.max(torch.abs(split_out - full_out)).item(), 1e-3)
        self.assertLess(
            torch.max(torch.abs(split_state - full_state)).item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
