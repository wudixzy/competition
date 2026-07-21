import ast
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "tests" / "bench_gdn_masked_neumann_prefill.py"


class MaskedNeumannStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = BENCHMARK.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(BENCHMARK))

    def test_published_algorithm_parameters_are_fixed(self):
        constants = {}
        for node in self.tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
            ):
                constants[node.targets[0].id] = node.value.value
        self.assertEqual(constants["CHUNK_SIZE"], 64)
        self.assertEqual(constants["NEUMANN_ORDER"], 3)
        self.assertEqual(constants["RESIDUAL_TERMS"], 8)
        self.assertEqual(constants["QUALIFICATION_LENGTH"], 4096)
        self.assertEqual(constants["MIN_INVERSE_SPEEDUP"], 2.0)
        self.assertEqual(constants["MIN_COMPLETE_SPEEDUP"], 1.5)
        self.assertEqual(constants["MAX_ABS_LIMIT"], 1.0e-3)
        self.assertEqual(constants["RELATIVE_L2_LIMIT"], 1.0e-5)

    def test_algorithm_parameters_are_not_cli_scan_knobs(self):
        self.assertNotIn("--chunk-size", self.source)
        self.assertNotIn("--neumann-order", self.source)
        self.assertNotIn("--residual-terms", self.source)
        function = next(
            node for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "masked_neumann_residual_inverse"
        )
        self.assertEqual(
            [argument.arg for argument in function.args.args],
            ["attn", "identity"],
        )

    def test_experiment_does_not_patch_production_runtime(self):
        production = (
            ROOT / "qwen3_6_scripts" / "qwen3_5.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("masked_neumann_residual_inverse", production)
        run_config = (ROOT / "computility-run.yaml").read_text(encoding="utf-8")
        self.assertNotIn("NEUMANN", run_config)


try:
    import torch
except Exception as exc:  # pragma: no cover - local env may lack torch
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"torch unavailable: {_TORCH_IMPORT_ERROR}")
class MaskedNeumannDynamicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "bench_gdn_masked_neumann_prefill", BENCHMARK)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.module = module

    def test_order_three_is_exact_for_four_by_four_input(self):
        torch.manual_seed(20260721)
        lower = torch.tril(torch.randn(2, 4, 4) * 0.05, diagonal=-1)
        identity = torch.eye(4)
        expected = torch.linalg.inv(identity - lower)
        actual = self.module.masked_neumann_residual_inverse(
            lower, identity)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLess(float((actual - expected).abs().max()), 1.0e-6)


if __name__ == "__main__":
    unittest.main()
