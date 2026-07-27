import base64
import importlib.util
import pathlib
import struct
import unittest
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "tests" / "probe_multimodal_preprocessing.py"
SPEC = importlib.util.spec_from_file_location(
    "probe_multimodal_preprocessing", PROBE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def decode_png(url: str) -> tuple[int, int, bytes]:
    prefix, encoded = url.split(",", 1)
    if prefix != "data:image/png;base64":
        raise AssertionError("unexpected data URL")
    image = base64.b64decode(encoded)
    if image[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("invalid PNG signature")
    offset = 8
    width = height = None
    payload = b""
    while offset < len(image):
        size = struct.unpack(">I", image[offset:offset + 4])[0]
        kind = image[offset + 4:offset + 8]
        data = image[offset + 8:offset + 8 + size]
        offset += 12 + size
        if kind == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
        elif kind == b"IDAT":
            payload += data
        elif kind == b"IEND":
            break
    assert width is not None and height is not None
    return width, height, zlib.decompress(payload)


class ProbeMultimodalPreprocessingUnitTest(unittest.TestCase):

    def test_synthetic_png_is_deterministic_and_color_sensitive(self):
        red_a = MODULE._solid_png_data_url((255, 0, 0))
        red_b = MODULE._solid_png_data_url((255, 0, 0))
        blue = MODULE._solid_png_data_url((0, 0, 255))
        self.assertEqual(red_a, red_b)
        self.assertNotEqual(red_a, blue)
        width, height, pixels = decode_png(red_a)
        self.assertEqual((width, height), (128, 128))
        self.assertEqual(pixels[:4], b"\x00\xff\x00\x00")

    def test_payload_preserves_image_order(self):
        first = MODULE._payload(["first", "second"])
        second = MODULE._payload(["second", "first"])
        first_urls = [
            item["image_url"]["url"]
            for item in first["messages"][0]["content"][:-1]
        ]
        second_urls = [
            item["image_url"]["url"]
            for item in second["messages"][0]["content"][:-1]
        ]
        self.assertEqual(first_urls, ["first", "second"])
        self.assertEqual(second_urls, ["second", "first"])

    def test_source_loads_tokenizer_but_not_model_weights(self):
        source = PROBE_PATH.read_text(encoding="utf-8")
        self.assertIn("AutoTokenizer.from_pretrained", source)
        self.assertNotIn("AutoModel", source)
        self.assertNotIn("load_weights", source)
        self.assertNotIn(".generate(", source)
        self.assertIn("contains_model_weights", source)


if __name__ == "__main__":
    unittest.main()
