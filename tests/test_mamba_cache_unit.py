import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAMBA_CACHE = ROOT / "qwen3_6_scripts" / "mamba_cache.py"


def _load_is_new_cache_entry():
    tree = ast.parse(MAMBA_CACHE.read_text(), filename=str(MAMBA_CACHE))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_is_new_cache_entry"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Dict": dict}
    exec(compile(module, str(MAMBA_CACHE), "exec"), namespace)
    return namespace["_is_new_cache_entry"]


class MambaCacheEntryTest(unittest.TestCase):

    def test_new_request_is_new_entry(self):
        is_new = _load_is_new_cache_entry()
        self.assertTrue(is_new({"old": {1: 0}}, "new", 1))

    def test_new_sequence_for_existing_request_is_new_entry(self):
        is_new = _load_is_new_cache_entry()
        self.assertTrue(is_new({"request": {1: 0}}, "request", 2))

    def test_subsequent_chunk_reuses_active_entry(self):
        is_new = _load_is_new_cache_entry()
        self.assertFalse(is_new({"request": {1: 0}}, "request", 1))


if __name__ == "__main__":
    unittest.main()
