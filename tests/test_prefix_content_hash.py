import ast
import hashlib
import os
import pathlib
import struct
import unittest
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import Mock, patch

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
        "Dict": dict,
        "Image": Image,
        "List": List,
        "Mapping": Mapping,
        "Optional": Optional,
        "PrefixHash": bytes,
        "Sequence": Any,
        "SequenceGroup": Any,
        "hashlib": hashlib,
        "logger": Mock(),
        "os": os,
        "struct": struct,
        "torch": None,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[target_class]


PrefixHash = _class_with_methods(
    PREFIX_BLOCK, "PrefixCachingBlock", {"hash_block_tokens"}, "PrefixHash")
NamespaceHash = _class_with_methods(
    BLOCK_MANAGER, "BlockSpaceManagerV2", {
        "_adapter_cache_namespace", "_build_runtime_cache_namespace",
        "_get_cache_namespace", "_sort_map_keys", "_hash_multi_modal_obj",
        "_hash_multi_modal_namespace", "_request_local_fallback_cache_namespace",
        "release_request_cache_namespace",
    }, "NamespaceHash")


class _FakeImage:

    def __init__(self, *, pixels: bytes, palette: list[int],
                 transparency: int, fail_read: bool = False):
        self.mode = "P"
        self.width = 2
        self.height = 2
        self._pixels = pixels
        self._palette = palette
        self._fail_read = fail_read
        self.palette = SimpleNamespace(mode="RGB")
        self.info = {"transparency": transparency}

    def tobytes(self):
        if self._fail_read:
            raise OSError("decode failure")
        return self._pixels

    def getpalette(self):
        return list(self._palette)


_FAKE_IMAGE_MODULE = SimpleNamespace(Image=_FakeImage)


def _all_fake_blocks(last_block):
    blocks = []
    while last_block is not None:
        blocks.append(last_block)
        last_block = last_block.prev_block
    return list(reversed(blocks))


def _load_allocator_fork():
    tree = ast.parse(PREFIX_BLOCK.read_text(), filename=str(PREFIX_BLOCK))
    source = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef)
                  and node.name == "PrefixCachingBlockAllocator")
    function = next(node for node in source.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "fork")
    namespace = {
        "Block": Any,
        "List": List,
        "get_all_blocks_recursively": _all_fake_blocks,
    }
    module = ast.fix_missing_locations(ast.Module(
        body=[function], type_ignores=[]))
    exec(compile(module, str(PREFIX_BLOCK), "exec"), namespace)
    return namespace["fork"]


def _load_allocator_init_block():
    tree = ast.parse(PREFIX_BLOCK.read_text(), filename=str(PREFIX_BLOCK))
    source = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef)
                  and node.name == "PrefixCachingBlockAllocator")
    function = next(node for node in source.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "_init_block")
    namespace = {"Block": Any, "List": List, "Optional": Optional}
    module = ast.fix_missing_locations(ast.Module(
        body=[function], type_ignores=[]))
    exec(compile(module, str(PREFIX_BLOCK), "exec"), namespace)
    return namespace["_init_block"]


class _FakeRefCounter:

    def incr(self, block_id):
        return 2


class _FakeForkBlock:

    def __init__(self, prev_block, token_ids, block_id, cache_namespace):
        self.prev_block = prev_block
        self.token_ids = token_ids
        self.block_id = block_id
        self._cache_namespace = cache_namespace
        self._cached_content_hash = None

    @property
    def cache_namespace(self):
        return self._cache_namespace

    @property
    def content_hash(self):
        if self._cached_content_hash is None:
            self._cached_content_hash = PrefixHash.hash_block_tokens(
                self.prev_block is None,
                None if self.prev_block is None else self.prev_block.content_hash,
                self.token_ids,
                self._cache_namespace,
            )
        return self._cached_content_hash


class _NamespaceResettingBlockPool:

    def init_block(self, prev_block, token_ids, block_size,
                   physical_block_id=None):
        return _FakeForkBlock(prev_block, token_ids, physical_block_id, b"")


class _FakeForkAllocator:
    fork = _load_allocator_fork()
    _init_block = _load_allocator_init_block()

    def __init__(self):
        self._block_size = 2
        self._refcounter = _FakeRefCounter()
        self._cache_namespace = None
        self._block_pool = _NamespaceResettingBlockPool()


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

    def test_palette_and_transparency_are_part_of_image_namespace(self):
        hasher = NamespaceHash()

        def palette_image(red: int, transparency: int):
            palette = [0] * 768
            palette[0:3] = [red, 20, 30]
            palette[3:6] = [40, 50, 60]
            return _FakeImage(
                pixels=b"\x00\x01\x00\x01",
                palette=palette,
                transparency=transparency)

        same_a = palette_image(10, 0)
        same_b = palette_image(10, 0)
        different_palette = palette_image(200, 0)
        different_transparency = palette_image(10, 1)
        image_globals = (
            NamespaceHash._hash_multi_modal_obj.__func__.__globals__)
        with patch.dict(image_globals, {"Image": _FAKE_IMAGE_MODULE}):
            self.assertEqual(
                hasher._hash_multi_modal_namespace({"image": same_a}),
                hasher._hash_multi_modal_namespace({"image": same_b}))
            self.assertNotEqual(
                hasher._hash_multi_modal_namespace({"image": same_a}),
                hasher._hash_multi_modal_namespace(
                    {"image": different_palette}))
            self.assertNotEqual(
                hasher._hash_multi_modal_namespace({"image": same_a}),
                hasher._hash_multi_modal_namespace(
                    {"image": different_transparency}))

    def test_unsupported_multimodal_value_fails_closed(self):
        instance = NamespaceHash()
        with self.assertRaises(TypeError):
            instance._hash_multi_modal_namespace({"bad": object()})
        instance._request_local_namespace = {}
        instance._runtime_cache_namespace = b"r" * 32
        first = instance._request_local_fallback_cache_namespace("request-a")
        self.assertEqual(
            first,
            instance._request_local_fallback_cache_namespace("request-a"))
        self.assertNotEqual(
            first,
            instance._request_local_fallback_cache_namespace("request-b"))

    def test_multimodal_hash_failure_uses_request_local_isolation(self):
        instance = NamespaceHash()
        instance._runtime_cache_namespace = b"r" * 32
        instance._request_local_namespace = {}
        instance._warned_mm_namespace_requests = set()
        sequence = SimpleNamespace(
            multi_modal_data={"image": _FakeImage(
                pixels=b"", palette=[0] * 768, transparency=0,
                fail_read=True)})
        seq_group = SimpleNamespace(
            lora_request=None, prompt_adapter_request=None)

        image_globals = (
            NamespaceHash._hash_multi_modal_obj.__func__.__globals__)
        with patch.dict(image_globals, {"Image": _FAKE_IMAGE_MODULE}):
            first = instance._get_cache_namespace(
                sequence, "request-a", seq_group)
            second = instance._get_cache_namespace(
                sequence, "request-a", seq_group)
        self.assertEqual(first, second)
        self.assertIn("request-a", instance._request_local_namespace)

    def test_multimodal_presence_does_not_evaluate_object_truthiness(self):
        class AmbiguousMultimodal:

            def __bool__(self):
                raise AssertionError("truthiness must not be evaluated")

        instance = NamespaceHash()
        instance._runtime_cache_namespace = b"r" * 32
        instance._request_local_namespace = {}
        instance._warned_mm_namespace_requests = set()
        sequence = SimpleNamespace(multi_modal_data=AmbiguousMultimodal())
        seq_group = SimpleNamespace(
            lora_request=None, prompt_adapter_request=None)
        namespace = instance._get_cache_namespace(
            sequence, "request-a", seq_group)
        self.assertEqual(len(namespace), 32)
        self.assertIn("request-a", instance._request_local_namespace)

    def test_request_local_namespace_is_fresh_after_release(self):
        instance = NamespaceHash()
        instance._runtime_cache_namespace = b"r" * 32
        instance._request_local_namespace = {}
        instance._warned_mm_namespace_requests = {"request-a"}
        with patch.object(os, "urandom",
                          side_effect=[b"a" * 32, b"b" * 32]):
            first = instance._request_local_fallback_cache_namespace(
                "request-a")
            self.assertEqual(
                first,
                instance._request_local_fallback_cache_namespace("request-a"))
            instance.release_request_cache_namespace("request-a")
            second = instance._request_local_fallback_cache_namespace(
                "request-a")
        self.assertNotEqual(first, second)
        self.assertNotIn("request-a", instance._warned_mm_namespace_requests)

    def test_runtime_namespace_covers_fixed_model_identity(self):
        instance = NamespaceHash()
        instance.block_size = 16
        with patch.dict(os.environ, {
                "BI100_PREFIX_MODEL_FINGERPRINT": "model-a",
                "BI100_PREFIX_DTYPE": "float16",
                "BI100_PREFIX_TP_SIZE": "4",
        }, clear=False):
            identity_a = instance._build_runtime_cache_namespace()
        with patch.dict(os.environ, {
                "BI100_PREFIX_MODEL_FINGERPRINT": "model-b",
                "BI100_PREFIX_DTYPE": "float16",
                "BI100_PREFIX_TP_SIZE": "4",
        }, clear=False):
            identity_b = instance._build_runtime_cache_namespace()
        self.assertEqual(len(identity_a), 32)
        self.assertNotEqual(identity_a, identity_b)

        with patch.dict(os.environ, {"BI100_PREFIX_TP_SIZE": "0"},
                        clear=False):
            with self.assertRaises(RuntimeError):
                instance._build_runtime_cache_namespace()

    def test_text_namespace_is_adapter_sensitive(self):
        instance = NamespaceHash()
        instance.block_size = 16
        instance._runtime_cache_namespace = b"r" * 32
        instance._request_local_namespace = {}
        instance._warned_mm_namespace_requests = set()
        sequence = SimpleNamespace(multi_modal_data=None)
        plain = SimpleNamespace(lora_request=None,
                                prompt_adapter_request=None)
        lora = SimpleNamespace(
            lora_request=SimpleNamespace(
                lora_name="adapter-a", lora_int_id=1,
                lora_path="/adapter-a", base_model_name="base"),
            prompt_adapter_request=None,
        )
        plain_namespace = instance._get_cache_namespace(
            sequence, "plain", plain)
        self.assertEqual(
            plain_namespace,
            instance._get_cache_namespace(sequence, "plain-2", plain))
        self.assertNotEqual(
            plain_namespace,
            instance._get_cache_namespace(sequence, "lora", lora))

    def test_initial_mutable_block_keeps_namespace(self):
        source = BLOCK_TABLE.read_text()
        self.assertIn("def _allocate_mutable_block", source)
        self.assertIn("allocate_mutable_block_with_cache_namespace", source)
        self.assertIn("block = self._allocate_mutable_block", source)

    def test_fork_preserves_first_block_namespace_and_hash_chain(self):
        namespace = b"request-namespace"
        first = _FakeForkBlock(None, [1, 2], 7, namespace)
        second = _FakeForkBlock(first, [3, 4], 8, namespace)

        forked = _FakeForkAllocator().fork(second)

        self.assertEqual([block.cache_namespace for block in forked],
                         [namespace, namespace])
        self.assertEqual([block.content_hash for block in forked],
                         [first.content_hash, second.content_hash])

    def test_physical_block_reuse_cannot_alias_logical_content(self):
        first = _FakeForkBlock(None, [1, 2], 7, b"namespace")
        same_content_new_physical = _FakeForkBlock(
            None, [1, 2], 99, b"namespace")
        reused_physical_new_content = _FakeForkBlock(
            None, [3, 4], 7, b"namespace")
        self.assertEqual(first.content_hash,
                         same_content_new_physical.content_hash)
        self.assertNotEqual(first.content_hash,
                            reused_physical_new_content.content_hash)


if __name__ == "__main__":
    unittest.main()
