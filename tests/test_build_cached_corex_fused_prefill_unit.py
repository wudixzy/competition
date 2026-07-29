from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_cached_corex_fused_prefill.py"
SPEC = importlib.util.spec_from_file_location("cached_corex_build", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CachedCorexBuildTest(unittest.TestCase):

    def test_cli_default_python_tracks_the_running_interpreter(self):
        self.assertEqual(
            Path(MODULE.sys.executable).resolve(),
            Path(sys.executable).resolve(),
        )
        self.assertIn(
            'default=Path(sys.executable)',
            SCRIPT.read_text(encoding="ascii"),
        )
        self.assertIn(
            '["/bin/bash", str(build_script), str(build_root)]',
            SCRIPT.read_text(encoding="ascii"),
        )

    def test_cached_artifact_name_matches_build_script_contract(self):
        build_script = (
            ROOT / "qwen3_6_scripts"
            / "build_corex_fused_paged_prefill_split4.sh"
        ).read_text(encoding="ascii")
        self.assertEqual(
            MODULE.ARTIFACT_NAME,
            "corex_fused_paged_prefill_split4.so",
        )
        self.assertIn(
            f"OUTPUT=${{VLLM_ROOT}}/{MODULE.ARTIFACT_NAME}",
            build_script,
        )

    def test_cache_key_changes_with_source_build_or_toolchain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "kernel.cu"
            build = root / "build.sh"
            source.write_text("source-a\n", encoding="ascii")
            build.write_text("build-a\n", encoding="ascii")
            toolchain = {
                "compiler_sha256": "a" * 64,
                "gpu_arch": "ivcore10",
            }
            first, _ = MODULE.cache_key(source, build, toolchain)
            source.write_text("source-b\n", encoding="ascii")
            second, _ = MODULE.cache_key(source, build, toolchain)
            source.write_text("source-a\n", encoding="ascii")
            build.write_text("build-b\n", encoding="ascii")
            third, _ = MODULE.cache_key(source, build, toolchain)
            changed_toolchain = dict(toolchain)
            changed_toolchain["compiler_sha256"] = "b" * 64
            fourth, _ = MODULE.cache_key(
                source, build, changed_toolchain)
        self.assertEqual(len({first, second, third, fourth}), 4)

    def test_cached_entry_rejects_artifact_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory)
            artifact = entry / MODULE.ARTIFACT_NAME
            artifact.write_bytes(b"artifact-a")
            inputs = {"source_sha256": "a" * 64}
            key = "b" * 64
            manifest = {
                "schema": MODULE.SCHEMA,
                "version": 1,
                "cache_key": key,
                "inputs": inputs,
                "build_succeeded": True,
                "artifact": {
                    "sha256": MODULE.sha256_file(artifact),
                    "size_bytes": artifact.stat().st_size,
                },
            }
            (entry / "manifest.json").write_text(
                __import__("json").dumps(manifest), encoding="ascii")
            valid, _ = MODULE._valid_cached_entry(
                entry, key, inputs)
            self.assertTrue(valid)
            artifact.write_bytes(b"artifact-b")
            valid, _ = MODULE._valid_cached_entry(
                entry, key, inputs)
            self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
