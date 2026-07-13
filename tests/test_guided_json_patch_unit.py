import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
PATCH_PATH = ROOT.parent / "qwen3_6_scripts" / "patch_xformers_sdpa_seq.py"
if not PATCH_PATH.exists():
    PATCH_PATH = ROOT / "staging_patch_xformers_sdpa_seq.py"
PATCH_SOURCE = PATCH_PATH.read_text()
JSON_STRING = re.compile(
    r'"(\\["\\/bfnrt]|\\u[0-9a-fA-F]{4}|[^"\\\x00-\x1f])*"')
JSON_WS = re.compile(r"[ \t\r\n]{1,4}")


class GuidedJsonPatchUnitTest(unittest.TestCase):

    def test_patch_installs_strict_json_string_terminal(self):
        self.assertIn("def patch_outlines_json_grammar(path):", PATCH_SOURCE)
        self.assertIn("JSON_STRING:", PATCH_SOURCE)
        self.assertIn("JSON_WS:", PATCH_SOURCE)
        self.assertIn("replace_one_of(", PATCH_SOURCE)
        self.assertIn(r'[^"\\\x00-\x1f]', PATCH_SOURCE)

    def test_json_string_rejects_raw_control_characters(self):
        self.assertIsNotNone(JSON_STRING.fullmatch(r'"escaped\nvalue"'))
        self.assertIsNotNone(JSON_STRING.fullmatch(r'"quote: \""'))
        for value in ('"raw\nvalue"', '"raw\tvalue"', '"raw\x01value"'):
            with self.subTest(value=repr(value)):
                self.assertIsNone(JSON_STRING.fullmatch(value))

    def test_json_whitespace_is_bounded(self):
        for value in (" ", "\n  \t", "\r\n"):
            self.assertIsNotNone(JSON_WS.fullmatch(value), repr(value))
        for value in ("", "     ", "\n\n\n\n\n"):
            self.assertIsNone(JSON_WS.fullmatch(value), repr(value))


if __name__ == "__main__":
    unittest.main()
