import importlib.util
import hashlib
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests/prefix_namespace_fork_gate.py"
SPEC = importlib.util.spec_from_file_location("prefix_fork_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _Block:

    def __init__(self, prev_block, token_ids, block_id, namespace,
                 content_hash=None):
        self.prev_block = prev_block
        self.token_ids = token_ids
        self.block_id = block_id
        self.cache_namespace = namespace
        parent = b"" if prev_block is None else prev_block.content_hash
        self.content_hash = content_hash or hashlib.sha256(
            namespace + parent + bytes(token_ids)).digest()


class _Allocator:

    def __init__(self, num_blocks, drop_namespace=False, fail_free=False,
                 stale_reuse=False):
        self.num_blocks = num_blocks
        self.drop_namespace = drop_namespace
        self.fail_free = fail_free
        self.stale_reuse = stale_reuse
        self._cached_blocks = {}
        self._hash_for_id = {}
        self._refcounts = {block_id: 0 for block_id in range(num_blocks)}
        self._computed = {block_id: False for block_id in range(num_blocks)}

    def _allocate(self, prev_block, token_ids, cache_namespace):
        parent = b"" if prev_block is None else prev_block.content_hash
        content_hash = hashlib.sha256(
            cache_namespace + parent + bytes(token_ids)).digest()
        if content_hash in self._cached_blocks:
            block_id = self._cached_blocks[content_hash]
        else:
            free_ids = [block_id for block_id, count in self._refcounts.items()
                        if count == 0]
            if not free_ids:
                raise AssertionError("no free blocks")
            block_id = free_ids[0]
            old_hash = self._hash_for_id.get(block_id)
            if old_hash is not None:
                self._cached_blocks.pop(old_hash, None)
            if self.stale_reuse and old_hash is not None:
                content_hash = old_hash
            self._cached_blocks[content_hash] = block_id
            self._hash_for_id[block_id] = content_hash
            self._computed[block_id] = bool(
                self.stale_reuse and old_hash is not None)
        self._refcounts[block_id] += 1
        return _Block(prev_block, token_ids, block_id, cache_namespace,
                      content_hash=content_hash)

    def allocate_immutable_blocks_with_cache_namespace(
            self, prev_block, block_token_ids, cache_namespace):
        blocks = []
        for token_ids in block_token_ids:
            prev_block = self._allocate(
                prev_block, token_ids, cache_namespace)
            blocks.append(prev_block)
        return blocks

    def allocate_immutable_block_with_cache_namespace(
            self, prev_block, token_ids, cache_namespace):
        return self._allocate(prev_block, token_ids, cache_namespace)

    def fork(self, last_block):
        source = []
        while last_block is not None:
            source.append(last_block)
            last_block = last_block.prev_block
        prev = None
        result = []
        for block in reversed(source):
            namespace = b"" if self.drop_namespace else block.cache_namespace
            self._refcounts[block.block_id] += 1
            prev = _Block(prev, block.token_ids, block.block_id, namespace,
                          content_hash=block.content_hash)
            result.append(prev)
        return result

    def free(self, block):
        if self.fail_free:
            raise AssertionError("release failed")
        self._refcounts[block.block_id] -= 1
        block.block_id = None

    def mark_blocks_as_computed(self, block_ids):
        for block_id in block_ids:
            self._computed[block_id] = True

    def block_is_computed(self, block_id):
        return self._computed[block_id]

    def get_num_free_blocks(self):
        return sum(count == 0 for count in self._refcounts.values())

    def get_num_total_blocks(self):
        return self.num_blocks


class PrefixNamespaceForkGateUnitTest(unittest.TestCase):

    def test_two_release_orders_qualify(self):
        report = MODULE.build_report(_Allocator)
        self.assertTrue(report["qualified"])
        self.assertEqual(len(report["cases"]), 2)
        self.assertTrue(report["physical_reuse"]["same_physical_block_id"])
        self.assertTrue(report["physical_reuse"]["content_hash_changed"])
        self.assertTrue(report["physical_reuse"]["old_hash_removed"])
        self.assertEqual(report["reasons"], [])

    def test_namespace_and_release_fail_closed(self):
        namespace = MODULE.build_report(
            lambda num_blocks: _Allocator(
                num_blocks, drop_namespace=True))
        self.assertFalse(namespace["qualified"])
        self.assertTrue(any("namespace mismatch" in reason
                            for reason in namespace["reasons"]))

        release = MODULE.build_report(lambda num_blocks: _Allocator(
            num_blocks, fail_free=True))
        self.assertFalse(release["qualified"])
        self.assertTrue(any("release failed" in reason
                            for reason in release["reasons"]))

    def test_stale_physical_reuse_fails_closed(self):
        report = MODULE.build_report(lambda num_blocks: _Allocator(
            num_blocks, stale_reuse=True))
        self.assertFalse(report["qualified"])
        self.assertTrue(any(
            "physical-reuse" in reason for reason in report["reasons"]))

    def test_atomic_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "gate.json"
            MODULE.atomic_write(path, {"qualified": True})
            self.assertIn('"qualified": true', path.read_text())


if __name__ == "__main__":
    unittest.main()
