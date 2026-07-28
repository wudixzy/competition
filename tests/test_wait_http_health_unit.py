#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import wait_http_health as health  # noqa: E402


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


class Response:
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b"ok"


class WaitHttpHealthUnitTest(unittest.TestCase):
    def test_deadline_is_absolute_and_request_timeout_shrinks(self) -> None:
        clock = FakeClock()
        observed_timeouts: list[float] = []

        def fail(_url: str, *, timeout: float) -> Response:
            observed_timeouts.append(timeout)
            raise TimeoutError

        report = health.wait_for_health(
            "http://127.0.0.1:8000/health",
            pid=123,
            starttime_ticks=456,
            timeout_s=2.5,
            opener=fail,
            process_starttime=lambda _pid: 456,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertFalse(report["qualified"])
        self.assertEqual(report["reason"], "deadline_expired")
        self.assertEqual(report["elapsed_s"], 2.5)
        self.assertEqual(observed_timeouts, [2.5, 1.5, 0.5])
        self.assertLessEqual(max(observed_timeouts), 2.5)

    def test_success_returns_immediately(self) -> None:
        clock = FakeClock()
        report = health.wait_for_health(
            "http://127.0.0.1:8000/health",
            pid=123,
            starttime_ticks=456,
            timeout_s=5,
            opener=lambda _url, timeout: Response(),
            process_starttime=lambda _pid: 456,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertTrue(report["qualified"])
        self.assertEqual(report["reason"], "healthy")
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(report["elapsed_s"], 0)

    def test_process_identity_loss_fails_without_waiting_to_deadline(self) -> None:
        clock = FakeClock()

        def fail(_url: str, *, timeout: float) -> Response:
            raise ConnectionError

        report = health.wait_for_health(
            "http://127.0.0.1:8000/health",
            pid=123,
            starttime_ticks=456,
            timeout_s=5,
            opener=fail,
            process_starttime=lambda _pid: 999,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertFalse(report["qualified"])
        self.assertEqual(report["reason"], "service_identity_lost")
        self.assertEqual(report["elapsed_s"], 0)

    def test_success_is_rejected_when_identity_changes_after_response(
            self) -> None:
        clock = FakeClock()
        observed = iter((456, 999))
        report = health.wait_for_health(
            "http://127.0.0.1:8000/health",
            pid=123,
            starttime_ticks=456,
            timeout_s=5,
            opener=lambda _url, timeout: Response(),
            process_starttime=lambda _pid: next(observed),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.assertFalse(report["qualified"])
        self.assertEqual(
            report["reason"], "service_identity_lost_after_health")
        self.assertEqual(report["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
