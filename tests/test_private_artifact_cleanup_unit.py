from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/lib/private_artifacts.sh"


class PrivateArtifactCleanupTests(unittest.TestCase):

    def test_removes_only_teacher_forced_private_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            for arm in ("control", "candidate"):
                arm_root = run_root / arm
                arm_root.mkdir()
                (arm_root / "teacher_forced_observation.json").write_text(
                    "private\n", encoding="utf-8")
                (arm_root / ".teacher_forced_observation.json.123.tmp").write_text(
                    "private\n", encoding="utf-8")
                (arm_root / "keep.json").write_text(
                    "keep\n", encoding="utf-8")
            command = (
                f'source "{HELPER}"\n'
                'remove_teacher_forced_observations "$1"'
            )
            result = subprocess.run(
                ["bash", "-c", command, "cleanup-test", str(run_root)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for arm in ("control", "candidate"):
                arm_root = run_root / arm
                self.assertFalse(
                    (arm_root / "teacher_forced_observation.json").exists())
                self.assertFalse(any(arm_root.glob(
                    ".teacher_forced_observation.json.*.tmp")))
                self.assertTrue((arm_root / "keep.json").is_file())

    def test_absent_private_files_are_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            (run_root / "control").mkdir()
            (run_root / "candidate").mkdir()
            command = (
                f'source "{HELPER}"\n'
                'remove_teacher_forced_observations "$1"'
            )
            result = subprocess.run(
                ["bash", "-c", command, "cleanup-test", str(run_root)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_preserves_callers_nullglob_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            (run_root / "control").mkdir()
            (run_root / "candidate").mkdir()
            command = (
                f'source "{HELPER}"\n'
                'shopt -s nullglob\n'
                'remove_teacher_forced_observations "$1"\n'
                'shopt -q nullglob'
            )
            result = subprocess.run(
                ["bash", "-c", command, "cleanup-test", str(run_root)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
