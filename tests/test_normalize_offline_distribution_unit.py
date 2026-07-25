from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "normalize_offline_distribution.py"
INSTALLER = ROOT / "scripts" / "install_bi100_bare_host_runtime.sh"
SPEC = importlib.util.spec_from_file_location(
    "normalize_offline_distribution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _metadata() -> str:
    return "Metadata-Version: 2.1\nName: transformers\nVersion: 4.55.3\n"


class NormalizeOfflineDistributionTest(unittest.TestCase):

    def _fixture(self, directory: str) -> tuple[Path, Path, Path]:
        root = Path(directory)
        site = root / "site-packages"
        dist_info = site / "transformers-4.55.3.dist-info"
        dist_info.mkdir(parents=True)
        (dist_info / "METADATA").write_text(
            _metadata(), encoding="utf-8")
        (dist_info / "direct_url.json").write_text(
            '{"url":"file:///tmp/bi100-patch-stage.ABC123/wheels/'
            'transformers-4.55.3-py3-none-any.whl"}\n',
            encoding="utf-8",
        )
        (dist_info / "RECORD").write_text(
            "transformers/__init__.py,sha256=old,1\n"
            "transformers-4.55.3.dist-info/direct_url.json,"
            "sha256=random,99\n"
            "transformers-4.55.3.dist-info/RECORD,,\n",
            encoding="utf-8",
        )
        wheel = root / "transformers-4.55.3-py3-none-any.whl"
        wheel.write_bytes(b"fixed offline wheel")
        return site, dist_info, wheel

    def test_normalization_is_canonical_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            site, dist_info, wheel = self._fixture(directory)
            report = MODULE.normalize_distribution(
                site_packages=site,
                distribution="Transformers",
                version="4.55.3",
                wheel=wheel,
            )
            direct_path = dist_info / "direct_url.json"
            record_path = dist_info / "RECORD"
            first_direct = direct_path.read_bytes()
            first_record = record_path.read_bytes()

            wheel_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            direct_url = json.loads(first_direct)
            self.assertEqual(
                direct_url["url"],
                "file:///offline/transformers-4.55.3-py3-none-any.whl",
            )
            self.assertEqual(
                direct_url["archive_info"]["hash"],
                f"sha256={wheel_digest}",
            )
            self.assertEqual(
                direct_url["archive_info"]["hashes"]["sha256"],
                wheel_digest,
            )
            self.assertNotIn(b"/tmp/", first_direct)

            rows = list(csv.reader(io.StringIO(first_record.decode("utf-8"))))
            direct_row = next(
                row for row in rows if row[0].endswith("direct_url.json"))
            expected_hash = base64.urlsafe_b64encode(
                hashlib.sha256(first_direct).digest()
            ).rstrip(b"=").decode("ascii")
            self.assertEqual(direct_row[1], f"sha256={expected_hash}")
            self.assertEqual(direct_row[2], str(len(first_direct)))
            self.assertEqual(report["wheel_sha256"], wheel_digest)

            MODULE.normalize_distribution(
                site_packages=site,
                distribution="transformers",
                version="4.55.3",
                wheel=wheel,
            )
            self.assertEqual(direct_path.read_bytes(), first_direct)
            self.assertEqual(record_path.read_bytes(), first_record)

    def test_missing_or_duplicate_distribution_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory) / "site-packages"
            site.mkdir()
            wheel = Path(directory) / "offline.whl"
            wheel.write_bytes(b"wheel")
            with self.assertRaisesRegex(ValueError, "found 0"):
                MODULE.normalize_distribution(
                    site_packages=site,
                    distribution="transformers",
                    version="4.55.3",
                    wheel=wheel,
                )

            for name in (
                "transformers-4.55.3.dist-info",
                "transformers-4.55.3-copy.dist-info",
            ):
                dist_info = site / name
                dist_info.mkdir()
                (dist_info / "METADATA").write_text(
                    _metadata(), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "found 2"):
                MODULE.normalize_distribution(
                    site_packages=site,
                    distribution="transformers",
                    version="4.55.3",
                    wheel=wheel,
                )

    def test_duplicate_record_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            site, dist_info, wheel = self._fixture(directory)
            record = dist_info / "RECORD"
            record.write_text(
                record.read_text(encoding="utf-8")
                + "transformers-4.55.3.dist-info/direct_url.json,,\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "found 2"):
                MODULE.normalize_distribution(
                    site_packages=site,
                    distribution="transformers",
                    version="4.55.3",
                    wheel=wheel,
                )

    def test_installer_help_and_unknown_option_are_side_effect_free(self):
        help_result = subprocess.run(
            ["bash", str(INSTALLER), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("Usage:", help_result.stdout)
        unknown_result = subprocess.run(
            ["bash", str(INSTALLER), "--not-a-real-option"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(unknown_result.returncode, 2)
        self.assertIn("unknown option", unknown_result.stderr)


if __name__ == "__main__":
    unittest.main()
