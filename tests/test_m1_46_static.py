from __future__ import annotations

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class M146StaticContractTest(unittest.TestCase):

    def test_cache_engine_keeps_default_paged_path(self):
        source = read("vllm/worker/cache_engine.py")
        self.assertIn("transfer_layout_from_env()", source)
        self.assertIn("Bi100BlockMajorKvTransfer", source)
        self.assertIn("self.attn_backend.swap_blocks", source)
        self.assertIn("self._bi100_block_major_transfer.swap_out", source)
        self.assertIn("self._bi100_block_major_transfer.swap_in", source)

    def test_docker_installs_authoritative_worker_modules(self):
        dockerfile = read("Dockerfile")
        patch_ops = read("qwen3_6_scripts/patch_ops.sh")
        for filename in ("bi100_block_major_kv.py", "cache_engine.py"):
            self.assertIn(
                f"vllm/worker/{filename} /workspace/qwen3_6_scripts/"
                f"vendor_overrides/vllm/worker/{filename}",
                dockerfile,
            )
            self.assertIn(
                f'"${{VLLM_OVERRIDE_ROOT}}/worker/{filename}"', patch_ops)

    def test_fixed_shape_and_chunk_are_not_yaml_tunables(self):
        module = read("vllm/worker/bi100_block_major_kv.py")
        yaml = read("computility-run.yaml")
        self.assertIn("STAGING_BLOCKS = 512", module)
        self.assertIn("EXPECTED_LAYERS = 10", module)
        self.assertIn("EXPECTED_BLOCK_PAYLOAD_ELEMENTS = 16 * 1 * 256", module)
        self.assertNotIn("BI100_CPU_KV_TRANSFER_LAYOUT", yaml)

    def test_corex_extension_exports_pack_and_scatter(self):
        source = read("qwen3_6_scripts/corex_block_major_kv_transfer.cu")
        build = read("qwen3_6_scripts/build_corex_block_major_kv_transfer.sh")
        self.assertIn('module.def("pack"', source)
        self.assertIn('module.def("scatter"', source)
        self.assertIn("corex_block_major_kv_transfer.so", build)
        self.assertIn("--cuda-gpu-arch=ivcore10", build)
        self.assertIn("block_ids.min().item<int64_t>()", source)
        self.assertIn("block_ids.max().item<int64_t>()", source)

    def test_single_staging_buffer_is_not_reused_asynchronously(self):
        source = read("vllm/worker/bi100_block_major_kv.py")
        scatter = source.index("self._extension.scatter")
        synchronize = source.index(
            "torch.cuda.current_stream(self.device).synchronize()", scatter)
        self.assertGreater(synchronize, scatter)


if __name__ == "__main__":
    unittest.main()
