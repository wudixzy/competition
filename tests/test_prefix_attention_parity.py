import importlib.util
import pathlib
import unittest

try:
    import torch
except ImportError:
    torch = None


ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGED_ATTN = ROOT / "qwen3_6_scripts" / "paged_attn.py"


def _load_paged_attention():
    spec = importlib.util.spec_from_file_location(
        "paged_attn_prefix_parity", PAGED_ATTN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PagedAttention


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PrefixAttentionParityTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.paged_attention = _load_paged_attention()
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest(f"vLLM/CoreX runtime unavailable: {exc}")

    def test_cached_suffix_matches_strictly_segmented_prefill(self):
        torch.manual_seed(20260712)
        token_count = 33
        block_size = 16
        num_q_heads = 2
        num_kv_heads = 1
        head_dim = 8

        query = torch.randn(token_count, num_q_heads, head_dim)
        key = torch.randn(token_count, num_kv_heads, head_dim)
        value = torch.randn(token_count, num_kv_heads, head_dim)

        num_blocks = (token_count + block_size - 1) // block_size
        padded_key = torch.zeros(
            num_blocks * block_size, num_kv_heads, head_dim)
        padded_value = torch.zeros_like(padded_key)
        padded_key[:token_count] = key
        padded_value[:token_count] = value
        key_cache = (padded_key.view(
            num_blocks, block_size, num_kv_heads, head_dim // 4, 4)
            .permute(0, 2, 3, 1, 4).contiguous())
        value_cache = (padded_value.view(
            num_blocks, block_size, num_kv_heads, head_dim)
            .permute(0, 2, 3, 1).contiguous())
        block_tables = torch.arange(num_blocks).view(1, -1)

        full_output = self.paged_attention._forward_prefix_pytorch(
            query,
            key,
            value,
            key_cache,
            value_cache,
            block_tables,
            torch.tensor([0, token_count]),
            torch.tensor([token_count]),
            torch.tensor([0]),
        )
        cached_output = self.paged_attention._forward_prefix_pytorch(
            query[-1:],
            key[-1:],
            value[-1:],
            key_cache,
            value_cache,
            block_tables,
            torch.tensor([0, 1]),
            torch.tensor([token_count]),
            torch.tensor([token_count - 1]),
        )

        torch.testing.assert_close(
            full_output[-1:], cached_output, rtol=0, atol=0)

        dense_outputs = []
        expanded_key = key.expand(-1, num_q_heads, -1)
        expanded_value = value.expand(-1, num_q_heads, -1)
        scale = head_dim ** -0.5
        for position in range(token_count):
            scores = torch.einsum(
                "hd,thd->ht", query[position] * scale,
                expanded_key[:position + 1])
            weights = torch.softmax(scores, dim=-1)
            dense_outputs.append(torch.einsum(
                "ht,thd->hd", weights,
                expanded_value[:position + 1]))
        dense_output = torch.stack(dense_outputs)
        torch.testing.assert_close(
            full_output, dense_output, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
