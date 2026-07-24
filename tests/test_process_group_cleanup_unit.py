import os
import pathlib
import subprocess
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "scripts/lib/process_group.sh"
ACTIVE_HARNESSES = (
    ROOT / "scripts/run_m1_49_hybrid_kv_ab.sh",
    ROOT / "scripts/run_m1_49_long_context_gates.sh",
    ROOT / "scripts/run_m1_48_prefill_profile.sh",
)


class ProcessGroupCleanupTest(unittest.TestCase):

    def run_with_fake_ps(self, table, command):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            ps = root / "ps"
            ps.write_text(textwrap.dedent(f"""\
                #!/bin/sh
                cat <<'EOF'
                {table}
                EOF
            """))
            ps.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{root}:{env['PATH']}"
            return subprocess.run(
                ["bash", "-c", f"source {LIBRARY!s}; {command}"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_zombie_only_group_has_no_live_members(self):
        table = "5307 Z\n5307 Zs"
        live = self.run_with_fake_ps(
            table, "test $(bi100_process_group_count 5307 live) -eq 0")
        zombies = self.run_with_fake_ps(
            table, "test $(bi100_process_group_count 5307 zombie) -eq 2")
        self.assertEqual(live.returncode, 0, live.stderr)
        self.assertEqual(zombies.returncode, 0, zombies.stderr)

    def test_live_states_fail_closed_while_zombies_are_separate(self):
        table = "5307 Z\n5307 Sl\n5307 D"
        live = self.run_with_fake_ps(
            table, "test $(bi100_process_group_count 5307 live) -eq 2")
        zombies = self.run_with_fake_ps(
            table, "test $(bi100_process_group_count 5307 zombie) -eq 1")
        self.assertEqual(live.returncode, 0, live.stderr)
        self.assertEqual(zombies.returncode, 0, zombies.stderr)

    def test_invalid_process_group_is_rejected(self):
        result = subprocess.run(
            ["bash", "-c", f"source {LIBRARY!s}; "
             "bi100_process_group_count 0 live"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)

    def test_real_process_group_is_terminated(self):
        process = subprocess.Popen(
            ["sleep", "60"],
            start_new_session=True,
        )
        try:
            pgid = os.getpgid(process.pid)
            result = subprocess.run(
                ["bash", "-c", f"source {LIBRARY!s}; "
                 f"bi100_stop_process_group {pgid} {process.pid} 3 3"],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            process.wait(timeout=3)
            self.assertIsNotNone(process.returncode)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)

    def test_active_harnesses_source_shared_cleanup(self):
        for harness in ACTIVE_HARNESSES:
            source = harness.read_text()
            self.assertIn(
                'source "$ROOT/scripts/lib/process_group.sh"', source)
            self.assertIn(
                'bi100_stop_process_group "$ACTIVE_PGID" "$ACTIVE_PID"',
                source,
            )
            self.assertNotIn('pgrep -g "$ACTIVE_PGID"', source)


if __name__ == "__main__":
    unittest.main()
