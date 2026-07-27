from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "qwen3_6_scripts/qwen3_5.py"


class GdnCombinedQkModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = MODEL.read_text(encoding="utf-8")

    def test_candidate_is_default_off_and_requires_qk_map(self):
        self.assertIn(
            'env_bool("BI100_GDN_COMBINED_QK_NORM", False)',
            self.source,
        )
        self.assertIn(
            "_USE_COREX_GDN_COMBINED_QK_NORM = (\n"
            "    _USE_COREX_GDN_QK_MAP",
            self.source,
        )

    def test_candidate_is_guarded_to_the_qualified_decode_shape(self):
        for guard in (
            "num_seqs == 1",
            "local_num_k == 4",
            "local_num_v == 8",
            "packed_mixed_qkv.dtype == torch.float16",
            "packed_mixed_qkv.shape == (1, 2048)",
            "packed_mixed_qkv.is_contiguous()",
        ):
            self.assertIn(guard, self.source)

    def test_candidate_combines_raw_qk_and_keeps_reference_fallback(self):
        self.assertIn(
            "packed_mixed_qkv.narrow(\n"
            "                            1, 0, 2 * local_key_dim).view(",
            self.source,
        )
        self.assertIn("normalized_qk = _l2norm(raw_qk)", self.source)
        self.assertIn(
            "normalized_q, normalized_k = torch.split(",
            self.source,
        )
        self.assertIn("normalized_q = _l2norm(q_raw)", self.source)
        self.assertIn("normalized_k = _l2norm(k_raw)", self.source)
        self.assertIn(
            "normalized_q, normalized_k, local_num_v",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
