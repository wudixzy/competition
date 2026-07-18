import ast
import pathlib
import typing
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = ROOT / "qwen3_6_scripts" / "qwen3_5.py"


def _load_helpers():
    tree = ast.parse(MODEL.read_text(), filename=str(MODEL))
    names = {"_gdn_segment_ends", "_validate_gdn_prefix_key"}
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": typing.Any,
        "Iterable": typing.Iterable,
        "List": typing.List,
        "Tuple": typing.Tuple,
        "RuntimeError": RuntimeError,
    }
    exec(compile(module, str(MODEL), "exec"), namespace)
    return namespace


class GdnPrefixCaptureTest(unittest.TestCase):

    def test_capture_boundary_is_forced_segment_end(self):
        segment_ends = _load_helpers()["_gdn_segment_ends"]
        self.assertEqual(
            segment_ends(8192, 4096, [8176]), [4096, 8176, 8192])
        self.assertEqual(segment_ends(3678, 4096, [1024, 3664]),
                         [1024, 3664, 3678])

    def test_query_end_capture_needs_no_extra_segment(self):
        segment_ends = _load_helpers()["_gdn_segment_ends"]
        self.assertEqual(segment_ends(8192, 4096, [8192]), [4096, 8192])

    def test_capture_offsets_are_deduplicated(self):
        segment_ends = _load_helpers()["_gdn_segment_ends"]
        self.assertEqual(segment_ends(5000, 4096, [1024, 1024]),
                         [1024, 4096, 5000])

    def test_stable_key_validation_fails_closed(self):
        validate = _load_helpers()["_validate_gdn_prefix_key"]
        key = (4, b"x" * 32)
        self.assertEqual(validate(key), key)
        for invalid in ((0, b"x" * 32), (1, b"short"), (1, 7), [1, b"x" * 32]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(RuntimeError):
                    validate(invalid)


if __name__ == "__main__":
    unittest.main()
