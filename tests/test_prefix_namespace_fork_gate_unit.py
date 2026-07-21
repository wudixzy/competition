import importlib.util
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

    def __init__(self, prev_block, token_ids, block_id, namespace):
        self.prev_block = prev_block
        self.token_ids = token_ids
        self.block_id = block_id
        self.cache_namespace = namespace
        parent = b"" if prev_block is None else prev_block.content_hash
        self.content_hash = parent + bytes(token_ids)


class _Allocator:

    def __init__(self, drop_namespace=False, fail_free=False):
        self.drop_namespace = drop_namespace
        self.fail_free = fail_free

    def allocate_immutable_blocks_with_cache_namespace(
            self, prev_block, block_token_ids, cache_namespace):
        blocks = []
        for index, token_ids in enumerate(block_token_ids):
            prev_block = _Block(prev_block, token_ids, index, cache_namespace)
            blocks.append(prev_block)
        return blocks

    def fork(self, last_block):
        source = []
        while last_block is not None:
            source.append(last_block)
            last_block = last_block.prev_block
        prev = None
        result = []
        for block in reversed(source):
            namespace = b"" if self.drop_namespace else block.cache_namespace
            prev = _Block(prev, block.token_ids, block.block_id, namespace)
            result.append(prev)
        return result

    def free(self, block):
        if self.fail_free:
            raise AssertionError("release failed")

    def get_num_free_blocks(self):
        return 8

    def get_num_total_blocks(self):
        return 8


class PrefixNamespaceForkGateUnitTest(unittest.TestCase):

    def test_two_release_orders_qualify(self):
        report = MODULE.build_report(_Allocator)
        self.assertTrue(report["qualified"])
        self.assertEqual(len(report["cases"]), 2)
        self.assertEqual(report["reasons"], [])

    def test_namespace_and_release_fail_closed(self):
        namespace = MODULE.build_report(
            lambda: _Allocator(drop_namespace=True))
        self.assertFalse(namespace["qualified"])
        self.assertTrue(any("namespace mismatch" in reason
                            for reason in namespace["reasons"]))

        release = MODULE.build_report(lambda: _Allocator(fail_free=True))
        self.assertFalse(release["qualified"])
        self.assertTrue(any("release failed" in reason
                            for reason in release["reasons"]))

    def test_atomic_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "gate.json"
            MODULE.atomic_write(path, {"qualified": True})
            self.assertIn('"qualified": true', path.read_text())


if __name__ == "__main__":
    unittest.main()
