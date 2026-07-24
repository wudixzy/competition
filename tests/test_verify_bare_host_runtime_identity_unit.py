from __future__ import annotations

import importlib.util
import hashlib
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "verify_bare_host_runtime_identity.py"
SPEC = importlib.util.spec_from_file_location("runtime_identity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BareHostRuntimeIdentityTest(unittest.TestCase):

    def build_fixture(self, root: pathlib.Path):
        source = root / "source"
        site = root / "runtime" / "site-packages"
        source.mkdir()
        site.mkdir(parents=True)
        files = {}
        for index, (name, relative) in enumerate(
                sorted(MODULE.DIRECT_SOURCE_FILES.items())):
            source_path = source / relative
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(f"direct-{index}\n".encode())
            installed_path = site / "installed" / f"{name}.py"
            installed_path.parent.mkdir(parents=True, exist_ok=True)
            installed_path.write_bytes(source_path.read_bytes())
            value = digest(source_path)
            files[name] = {
                "source_sha256": value,
                "installed_sha256": value,
                "same": True,
                "installed_path": str(installed_path),
            }
        for index, name in enumerate(sorted(MODULE.GENERATED_FILES)):
            installed_path = site / "generated" / f"{name}.py"
            installed_path.parent.mkdir(parents=True, exist_ok=True)
            installed_path.write_bytes(f"generated-{index}\n".encode())
            value = digest(installed_path)
            files[name] = {
                "source_sha256": value,
                "installed_sha256": value,
                "same": True,
                "installed_path": str(installed_path),
            }
        fixed = {
            "block_manager_base_sha256": source
            / "vllm/core/block_manager_v2.py",
            "cache_trace_patcher_sha256": source
            / "qwen3_6_scripts/patch_block_manager_cache_trace.py",
            "installer_sha256": source
            / "scripts/install_bi100_bare_host_runtime.sh",
        }
        for index, path in enumerate(fixed.values()):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixed-{index}\n".encode())
        report = {
            "schema": MODULE.INSTALL_SCHEMA,
            "version": 2,
            "qualified": True,
            "site_packages": str(site),
            "system_site_packages_modified": False,
            "source_revision": "a" * 40,
            "source_tree_clean": True,
            "versions": {"transformers": "4.55.3"},
            "files": files,
        }
        for field, path in fixed.items():
            report[field] = digest(path)
        return source, site, report

    def test_qualified_overlay_is_bound_to_current_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source, site, install = self.build_fixture(
                pathlib.Path(directory))
            report = MODULE.verify(source, site, install, "a" * 40)
        self.assertTrue(report["qualified"], report)
        self.assertEqual(report["reasons"], [])
        self.assertTrue(all(
            row["same"] for row in report["files"].values()))

    def test_revision_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, site, install = self.build_fixture(
                pathlib.Path(directory))
            report = MODULE.verify(source, site, install, "b" * 40)
        self.assertFalse(report["qualified"])
        self.assertIn(
            "runtime install revision differs from current source",
            report["reasons"],
        )

    def test_runtime_or_current_source_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, site, install = self.build_fixture(
                pathlib.Path(directory))
            direct = next(iter(MODULE.DIRECT_SOURCE_FILES.values()))
            (source / direct).write_text("changed\n", encoding="utf-8")
            generated = install["files"]["cache_trace_outputs"]
            pathlib.Path(generated["installed_path"]).write_text(
                "changed\n", encoding="utf-8")
            report = MODULE.verify(source, site, install, "a" * 40)
        self.assertFalse(report["qualified"])
        self.assertTrue(any(
            "runtime/current source identity differs" in reason
            for reason in report["reasons"]))


if __name__ == "__main__":
    unittest.main()
