import ast
import hashlib
import pathlib
import struct
import unittest
from collections.abc import Mapping
from typing import Any, List, Optional

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None  # type: ignore


ROOT = pathlib.Path(__file__).resolve().parents[1]
PREFIX_BLOCK = ROOT / "vllm/core/block/prefix_caching_block.py"
BLOCK_MANAGER = ROOT / "vllm/core/block_manager_v2.py"
BLOCK_TABLE = ROOT / "vllm/core/block/block_table.py"


def _class_with_methods(path: pathlib.Path, source_class: str,
                        method_names: set[str], target_class: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    source = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef)
                  and node.name == source_class)
    methods = [node for node in source.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name in method_names]
    cls = ast.ClassDef(name=target_class,
                       bases=[],
                       keywords=[],
                       body=methods,
                       decorator_list=[])
    module = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Image": Image,
        "List": List,
        "Mapping": Mapping,
        "Optional": Optional,
        "PrefixHash": bytes,
        "hashlib": hashlib,
        "struct": struct,
        "torch": None,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[target_class]


PrefixHash = _class_with_methods(
    PREFIX_BLOCK, "PrefixCachingBlock", {"hash_block_tokens"}, "PrefixHash")
NamespaceHash = _class_with_methods(
    BLOCK_MANAGER, "BlockSpaceManagerV2", {
        "_sort_map_keys", "_hash_multi_modal_obj",
        "_hash_multi_modal_namespace", "_request_local_fallback_cache_namespace",
    }, "NamespaceHash")


class PrefixContentHashTest(unittest.TestCase):

    def test_sha256_chain_is_stable_and_parent_sensitive(self):
        first = PrefixHash.hash_block_tokens(
            True, None, [1, 2, 3, 4], b"stable-namespace")
        first_again = PrefixHash.hash_block_tokens(
            True, None, [1, 2, 3, 4], b"stable-namespace")
        variant = PrefixHash.hash_block_tokens(
            True, None, [9, 2, 3, 4], b"stable-namespace")
        second = PrefixHash.hash_block_tokens(
            False, first, [5, 6, 7, 8])
        second_variant = PrefixHash.hash_block_tokens(
            False, variant, [5, 6, 7, 8])
        self.assertEqual(first, first_again)
        self.assertEqual(len(first), 32)
        self.assertNotEqual(first, variant)
        self.assertNotEqual(second, second_variant)

    def test_namespace_changes_first_and_downstream_hashes(self):
        first_a = PrefixHash.hash_block_tokens(True, None, [1, 2], b"a")
        first_b = PrefixHash.hash_block_tokens(True, None, [1, 2], b"b")
        self.assertNotEqual(first_a, first_b)
        self.assertNotEqual(
            PrefixHash.hash_block_tokens(False, first_a, [3, 4]),
            PrefixHash.hash_block_tokens(False, first_b, [3, 4]))

    def test_nested_multimodal_namespace_is_canonical(self):
        hasher = NamespaceHash()
        value_a = {"meta": {"scores": [1, 2, 3], "blob": b"abc"},
                   "temperature": 0.1}
        value_b = {"temperature": 0.1,
                   "meta": {"blob": b"abc", "scores": [1, 2, 3]}}
        self.assertEqual(hasher._hash_multi_modal_namespace(value_a),
                         hasher._hash_multi_modal_namespace(value_b))
        self.assertEqual(
            len(hasher._hash_multi_modal_namespace(value_a)), 32)

    def test_image_namespace_is_content_sensitive(self):
        if Image is None:
            self.skipTest("PIL is unavailable")
        hasher = NamespaceHash()
        same_a = Image.new("RGB", (2, 2), color=(10, 20, 30))
        same_b = Image.new("RGB", (2, 2), color=(10, 20, 30))
        different = Image.new("RGB", (2, 2), color=(200, 20, 30))
        self.assertEqual(
            hasher._hash_multi_modal_namespace({"image": same_a}),
            hasher._hash_multi_modal_namespace({"image": same_b}))
        self.assertNotEqual(
            hasher._hash_multi_modal_namespace({"image": same_a}),
            hasher._hash_multi_modal_namespace({"image": different}))

    def test_unsupported_multimodal_value_fails_closed(self):
        instance = NamespaceHash()
        with self.assertRaises(TypeError):
            instance._hash_multi_modal_namespace({"bad": object()})
        instance._request_local_namespace = {}
        first = instance._request_local_fallback_cache_namespace("request-a")
        self.assertEqual(
            first,
            instance._request_local_fallback_cache_namespace("request-a"))
        self.assertNotEqual(
            first,
            instance._request_local_fallback_cache_namespace("request-b"))

    def test_initial_mutable_block_keeps_namespace(self):
        source = BLOCK_TABLE.read_text()
        self.assertIn("def _allocate_mutable_block", source)
        self.assertIn("allocate_mutable_block_with_cache_namespace", source)
        self.assertIn("block = self._allocate_mutable_block", source)


if __name__ == "__main__":
    unittest.main()
