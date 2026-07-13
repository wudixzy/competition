import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_SOURCE = ROOT / "qwen3_6_scripts" / "qwen3_5.py"


class _FiniteResult:

    def __init__(self, calls):
        self.calls = calls

    def all(self):
        self.calls.append("all")
        return True


class _FakeTorch:
    Tensor = object

    def __init__(self, calls):
        self.calls = calls

    def isfinite(self, _tensor):
        self.calls.append("isfinite")
        return _FiniteResult(self.calls)


def _load_finite_check(enabled):
    tree = ast.parse(MODEL_SOURCE.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_check_gdn_finite"
    )
    calls = []
    namespace = {
        "torch": _FakeTorch(calls),
        "_GDN_FINITE_CHECK": enabled,
        "_ALLOW_GDN_NAN_ZERO": False,
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(MODEL_SOURCE), "exec"), namespace)
    return namespace["_check_gdn_finite"], calls


class GdnFiniteModeTest(unittest.TestCase):

    def test_disabled_mode_returns_without_touching_device_tensor(self):
        check, calls = _load_finite_check(False)
        tensor = object()
        self.assertIs(check(tensor, layer_idx=0, stage="decode"), tensor)
        self.assertEqual(calls, [])

    def test_enabled_mode_executes_finite_reduction(self):
        check, calls = _load_finite_check(True)
        tensor = object()
        self.assertIs(check(tensor, layer_idx=0, stage="decode"), tensor)
        self.assertEqual(calls, ["isfinite", "all"])


if __name__ == "__main__":
    unittest.main()
