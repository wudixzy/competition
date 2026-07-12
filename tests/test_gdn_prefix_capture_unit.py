import ast
import pathlib
import typing
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = ROOT / "qwen3_6_scripts" / "qwen3_5.py"


def _load_helpers():
    tree = ast.parse(MODEL.read_text(), filename=str(MODEL))
    names = {"_gdn_capture_offset", "_gdn_segment_ends"}
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"List": typing.List, "Optional": typing.Optional}
    exec(compile(module, str(MODEL), "exec"), namespace)
    return namespace


class GdnPrefixCaptureTest(unittest.TestCase):

    def test_unaligned_prompt_captures_last_complete_block(self):
        capture = _load_helpers()["_gdn_capture_offset"]
        self.assertEqual(capture(0, 3678, 16), 3664)

    def test_full_chunk_captures_strict_prefix(self):
        capture = _load_helpers()["_gdn_capture_offset"]
        self.assertEqual(capture(0, 8192, 16), 8176)
        self.assertEqual(capture(8192, 520, 16), 512)

    def test_no_new_complete_block_has_no_capture(self):
        capture = _load_helpers()["_gdn_capture_offset"]
        self.assertIsNone(capture(8176, 16, 16))
        self.assertIsNone(capture(8192, 8, 16))

    def test_capture_boundary_is_forced_segment_end(self):
        segment_ends = _load_helpers()["_gdn_segment_ends"]
        self.assertEqual(
            segment_ends(8192, 4096, 8176), [4096, 8176, 8192])
        self.assertEqual(segment_ends(3678, 4096, 3664), [3664, 3678])


if __name__ == "__main__":
    unittest.main()
