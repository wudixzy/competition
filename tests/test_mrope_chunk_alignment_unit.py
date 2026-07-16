import ast
import binascii
import importlib.util
import pathlib
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "qwen3_6_scripts" / "patch_model_runner.py"
API_PROBE_PATH = ROOT / "tests" / "mrope_chunk_api.py"
MODEL_RUNNER = ROOT / "vllm" / "worker" / "model_runner.py"
sys.path.insert(0, str(PATCH_PATH.parent))


def _load_patch_module():
    spec = importlib.util.spec_from_file_location(
        "qwen36_patch_model_runner_unit", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_api_probe_module():
    spec = importlib.util.spec_from_file_location(
        "qwen36_mrope_chunk_api_unit", API_PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_slice_helper(source: str):
    tree = ast.parse(source)
    helper = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_slice_mrope_positions")
    module = ast.Module(body=[helper], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "patched_model_runner.py", "exec"), namespace)
    return namespace["_slice_mrope_positions"]


class MropeChunkAlignmentTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.patch = _load_patch_module()

    def apply_patch(self) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "model_runner.py"
            target.write_text(MODEL_RUNNER.read_text())
            self.patch.patch_model_runner(target)
            return target.read_text()

    def test_chunked_multimodal_positions_match_current_query(self):
        source = self.apply_patch()
        helper = _load_slice_helper(source)
        positions = [list(range(26540)) for _ in range(3)]
        aligned = helper(positions, 26476, 26540, 64)
        self.assertEqual([len(axis) for axis in aligned], [64, 64, 64])
        self.assertEqual(aligned[0][0], 26476)
        self.assertEqual(aligned[2][-1], 26539)

    def test_partial_prefix_hit_crops_existing_mrope_positions(self):
        source = self.apply_patch()
        self.assertIn(
            "positions, uncomputed_start, None,", source)
        helper = _load_slice_helper(source)
        positions = [list(range(8192)) for _ in range(3)]
        aligned = helper(positions, 8128, None, 64)
        self.assertEqual([len(axis) for axis in aligned], [64, 64, 64])

    def test_full_prefix_hit_retains_only_last_mrope_position(self):
        source = self.apply_patch()
        self.assertIn(
            "_slice_mrope_positions(positions, -1, None, 1)", source)
        helper = _load_slice_helper(source)
        positions = [list(range(128)) for _ in range(3)]
        self.assertEqual(helper(positions, -1, None, 1),
                         [[127], [127], [127]])

    def test_alignment_mismatch_fails_before_gpu_execution(self):
        source = self.apply_patch()
        helper = _load_slice_helper(source)
        with self.assertRaisesRegex(RuntimeError, "length mismatch"):
            helper([list(range(10)) for _ in range(3)], 3, 8, 6)

    def test_api_probe_png_has_valid_chunks_and_payload(self):
        data = _load_api_probe_module()._TEST_PNG
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        offset = 8
        compressed = bytearray()
        while offset < len(data):
            length = struct.unpack(">I", data[offset:offset + 4])[0]
            chunk_type = data[offset + 4:offset + 8]
            payload = data[offset + 8:offset + 8 + length]
            expected_crc = struct.unpack(
                ">I", data[offset + 8 + length:offset + 12 + length])[0]
            self.assertEqual(
                binascii.crc32(chunk_type + payload) & 0xffffffff,
                expected_crc,
            )
            if chunk_type == b"IDAT":
                compressed.extend(payload)
            offset += 12 + length
        self.assertEqual(zlib.decompress(compressed), b"\x01\x00\xff")

    def test_patch_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp) / "model_runner.py"
            target.write_text(MODEL_RUNNER.read_text())
            self.patch.patch_model_runner(target)
            first = target.read_text()
            self.patch.patch_model_runner(target)
            self.assertEqual(target.read_text(), first)


if __name__ == "__main__":
    unittest.main()
