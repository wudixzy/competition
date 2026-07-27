import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "tests" / "probe_multimodal_limits.py"
SPEC = importlib.util.spec_from_file_location(
    "probe_multimodal_limits", PROBE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ProbeMultimodalLimitsUnitTest(unittest.TestCase):

    def test_source_is_weight_free_and_privacy_bounded(self):
        source = PROBE_PATH.read_text(encoding="utf-8")
        self.assertIn("skip_tokenizer_init", source)
        self.assertNotIn("from_pretrained", source)
        self.assertNotIn("data:image", source)
        self.assertNotIn("http://", source)
        self.assertIn("contains_image_url_or_bytes", source)

    def test_tracker_observation_classifies_only_fixed_limit_error(self):
        class Tracker:

            def __init__(self, _config, _tokenizer):
                self.items = []

            def add(self, _modality, item):
                if self.items:
                    raise ValueError(
                        "At most 1 image(s) may be provided in one request.")
                self.items.append(item)

            def all_mm_data(self):
                return {"image": self.items[0]["image"]}

        original_import = __import__

        def importing(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "vllm.entrypoints.chat_utils":
                class Fake:
                    MultiModalItemTracker = Tracker
                return Fake()
            return original_import(
                name, globals, locals, fromlist, level)

        import builtins
        saved = builtins.__import__
        builtins.__import__ = importing
        try:
            observation = MODULE._tracker_observation(object(), 2)
        finally:
            builtins.__import__ = saved

        self.assertEqual(observation, {
            "attempted": 2,
            "accepted": 1,
            "combined_count": 1,
            "reason": "image_count_limit",
            "error_type": "ValueError",
        })


if __name__ == "__main__":
    unittest.main()
