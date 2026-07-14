import ast
import pathlib
import types
import unittest
from functools import lru_cache

try:
    import torch
except Exception as exc:  # pragma: no cover - local environment may lack torch
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


ROOT = pathlib.Path(__file__).resolve().parents[1]
QWEN35 = ROOT / "qwen3_6_scripts" / "qwen3_5.py"


def _load_helpers():
    tree = ast.parse(QWEN35.read_text(), filename=str(QWEN35))
    names = {"_gdn_solve_identity", "_solve_gdn_lower_triangular"}
    helpers = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=helpers, type_ignores=[])
    ast.fix_missing_locations(module)
    ixformer_functions = types.SimpleNamespace(solve=torch.linalg.solve)
    namespace = {
        "torch": torch,
        "lru_cache": lru_cache,
        "ixformer_functions": ixformer_functions,
    }
    exec(compile(module, str(QWEN35), "exec"), namespace)
    return (namespace["_gdn_solve_identity"],
            namespace["_solve_gdn_lower_triangular"])


def _reference_inverse(attn):
    result = attn.clone()
    chunk_size = result.shape[-1]
    for index in range(1, chunk_size):
        row = result[..., index, :index].clone()
        sub = result[..., :index, :index].clone()
        result[..., index, :index] = (
            row + (row.unsqueeze(-1) * sub).sum(-2))
    return result + torch.eye(chunk_size, dtype=result.dtype)


@unittest.skipIf(torch is None, f"torch unavailable: {_TORCH_IMPORT_ERROR}")
class GdnExactSolveTest(unittest.TestCase):
    def setUp(self):
        self.identity, self.solve = _load_helpers()

    def test_refined_solve_matches_reference_recurrence(self):
        torch.manual_seed(20260715)
        attn = torch.tril(torch.randn(2, 3, 4, 8, 8) * 0.01, diagonal=-1)
        actual = self.solve(attn)
        expected = _reference_inverse(attn)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_identity_is_contiguous_and_cached(self):
        first = self.identity(12, 64, torch.device("cpu"))
        second = self.identity(12, 64, torch.device("cpu"))
        self.assertTrue(first.is_contiguous())
        self.assertEqual(first.shape, (12, 64, 64))
        self.assertEqual(first.data_ptr(), second.data_ptr())

    def test_non_square_input_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "expects square matrices"):
            self.solve(torch.zeros(2, 3, 4))


if __name__ == "__main__":
    unittest.main()
