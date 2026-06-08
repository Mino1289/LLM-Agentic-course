"""Red tests for the embedding pipeline backoff + quota state (PRD etape 3)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.embedding_pipeline import (
    BackoffConfig,
    QuotaState,
    with_exponential_backoff,
)


class _FakePermanentError(Exception):
    """Stand-in for a non-retryable error (auth, dim mismatch, ...)."""


class _FakeTransientError(Exception):
    """Stand-in for a retryable error (timeout, 429, network)."""


class BackoffRetryTests(unittest.TestCase):
    def test_succeeds_after_transient_retries(self) -> None:
        sleeps: list[float] = []

        def fake_sleep(value: float) -> None:
            sleeps.append(value)

        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise _FakeTransientError(f"boom {attempts['n']}")
            return "ok"

        result = with_exponential_backoff(
            flaky,
            config=BackoffConfig(max_retries=4, base_sec=1.0, cap_sec=60.0, jitter="full"),
            sleep=fake_sleep,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(attempts["n"], 3)
        # Two transient failures => two sleeps, each in [0, min(cap, base*2^attempt)].
        self.assertEqual(len(sleeps), 2)
        for attempt, slept in enumerate(sleeps, start=1):
            self.assertGreaterEqual(slept, 0.0)
            self.assertLessEqual(slept, min(60.0, 1.0 * (2 ** attempt)))

    def test_gives_up_after_max_retries(self) -> None:
        def always_fail() -> None:
            raise _FakeTransientError("nope")

        with self.assertRaises(_FakeTransientError):
            with_exponential_backoff(
                always_fail,
                config=BackoffConfig(max_retries=2, base_sec=0.0, cap_sec=1.0, jitter="none"),
                sleep=lambda _value: None,
            )

    def test_does_not_retry_on_dim_mismatch(self) -> None:
        attempts = {"n": 0}

        def fail_dim() -> None:
            attempts["n"] += 1
            raise ValueError("embedding dim mismatch: expected 1536 got 3072")

        with self.assertRaises(ValueError):
            with_exponential_backoff(
                fail_dim,
                config=BackoffConfig(max_retries=5, base_sec=0.0, cap_sec=1.0, jitter="none"),
                sleep=lambda _value: None,
            )

        # Permanent error: must fail-fast (one call only).
        self.assertEqual(attempts["n"], 1)

    def test_does_not_retry_on_auth_error(self) -> None:
        attempts = {"n": 0}

        def fail_auth() -> None:
            attempts["n"] += 1
            raise _FakePermanentError("401 Unauthorized: invalid api key")

        with self.assertRaises(_FakePermanentError):
            with_exponential_backoff(
                fail_auth,
                config=BackoffConfig(max_retries=5, base_sec=0.0, cap_sec=1.0, jitter="none"),
                sleep=lambda _value: None,
            )

        self.assertEqual(attempts["n"], 1)

    def test_full_jitter_sleeps_in_bounds(self) -> None:
        sleeps: list[float] = []

        def collect_sleep(value: float) -> None:
            sleeps.append(value)

        attempts = {"n": 0}

        def always_fail() -> None:
            attempts["n"] += 1
            raise _FakeTransientError("retry")

        with self.assertRaises(_FakeTransientError):
            with_exponential_backoff(
                always_fail,
                config=BackoffConfig(max_retries=6, base_sec=1.0, cap_sec=8.0, jitter="full"),
                sleep=collect_sleep,
            )

        # 6 attempts => 5 sleeps (no sleep after the last attempt).
        self.assertEqual(len(sleeps), 5)
        for attempt, slept in enumerate(sleeps, start=1):
            self.assertGreaterEqual(slept, 0.0)
            self.assertLessEqual(slept, min(8.0, 1.0 * (2 ** attempt)))


class QuotaStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "quota_state.json"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_persists_across_instances(self) -> None:
        first = QuotaState(self.path)
        first.update(batch_size=32, last_error=None)

        second = QuotaState(self.path)
        self.assertEqual(second.quota_used(), 32)
        payload = second.load()
        self.assertEqual(payload["quota_used"], 32)
        self.assertEqual(payload["last_batch_size"], 32)
        self.assertIsNone(payload["last_error"])
        self.assertIn("last_updated", payload)

    def test_update_increments_and_records_error(self) -> None:
        state = QuotaState(self.path)
        state.update(batch_size=10)
        state.update(batch_size=5, last_error="timeout")
        state.update(batch_size=7)

        self.assertEqual(state.quota_used(), 22)
        payload = state.load()
        self.assertEqual(payload["quota_used"], 22)
        self.assertEqual(payload["last_batch_size"], 7)
        self.assertEqual(payload["last_error"], "timeout")

    def test_load_returns_defaults_when_file_missing(self) -> None:
        state = QuotaState(self.path)
        self.assertEqual(state.quota_used(), 0)
        self.assertEqual(state.load()["quota_used"], 0)


class QuotaStateAtomicWriteTests(unittest.TestCase):
    """QuotaState.save() must write atomically (.tmp + rename).

    Without atomic writes, a process crash mid-write leaves the JSON
    file half-written, and the next ``load()`` either returns defaults
    (silent quota reset) or raises (operator-visible failure). The fix
    is the standard write-temp-then-rename pattern, identical to
    ``EmbeddingCache._save()``.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "quota_state.json"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_save_leaves_no_temp_file_behind(self) -> None:
        state = QuotaState(self.path)
        state.update(batch_size=10)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self.assertFalse(
            tmp.exists(),
            f"Atomic save must not leave a .tmp file behind ({tmp})",
        )

    def test_save_overwrite_does_not_corrupt_existing_file(self) -> None:
        """Simulate a crash: a stale .tmp is dropped before the rename.

        The existing ``quota_state.json`` must remain readable and
        unchanged. Without atomic writes, the next ``load()`` would
        either crash (partial JSON) or silently reset to defaults.
        """
        first = QuotaState(self.path)
        first.update(batch_size=42, last_error=None)
        self.assertTrue(self.path.exists())
        original_bytes = self.path.read_bytes()

        # Simulate a crashed prior save: drop a .tmp that has partial
        # garbage content. A non-atomic save() would happily leave this
        # on disk; an atomic save() must use it as a transient buffer
        # only and must NOT let a future load() see partial content.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text('{"date": "2026-06-06", "quota_us', encoding="utf-8")

        # Now do a fresh save. After this, the file must contain the
        # full JSON, the .tmp must be gone, and load() must succeed.
        # ``update`` is INCREMENTAL, so 42 + 100 = 142.
        second = QuotaState(self.path)
        second.update(batch_size=100)
        self.assertFalse(
            tmp.exists(),
            "Atomic save must consume the .tmp via Path.replace()",
        )
        payload = second.load()
        self.assertEqual(payload["quota_used"], 142)
        # The original file's content was 42; after the new save it
        # must reflect 142. (We don't insist on bytes-equal because the
        # timestamp changes; the point is: full, parseable JSON.)
        self.assertNotEqual(original_bytes, self.path.read_bytes())

    def test_save_then_load_round_trip(self) -> None:
        """Sanity: a save followed by a load returns the same payload."""
        state = QuotaState(self.path)
        state.update(batch_size=7, last_error="transient_429")
        fresh = QuotaState(self.path)
        payload = fresh.load()
        self.assertEqual(payload["quota_used"], 7)
        self.assertEqual(payload["last_batch_size"], 7)
        self.assertEqual(payload["last_error"], "transient_429")


class YFinanceRetryTests(unittest.TestCase):
    def test_download_with_retry_recovers_from_transient_error(self) -> None:
        # We import the wrapper function lazily to avoid pulling yfinance at import time.
        from rag.download_share_prices import download_with_retry

        sleeps: list[float] = []
        calls = {"n": 0}

        def fake_yf_download(ticker: str, start: str, end: str):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("yahoo throttle")
            df = MagicMock()
            df.empty = False
            return df

        with patch("rag.download_share_prices.yf.download", side_effect=fake_yf_download):
            with patch(
                "rag.embedding_pipeline.time.sleep", side_effect=lambda v: sleeps.append(v)
            ):
                df = download_with_retry("NVDA", "2024-01-01", "2024-12-31")

        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(sleeps), 1)

    def test_download_with_retry_gives_up_after_max(self) -> None:
        from rag.download_share_prices import download_with_retry

        def always_fail(ticker: str, start: str, end: str):
            raise ConnectionError("nope")

        with patch("rag.download_share_prices.yf.download", side_effect=always_fail):
            with patch("rag.embedding_pipeline.time.sleep", lambda _v: None):
                with self.assertRaises(ConnectionError):
                    download_with_retry("NVDA", "2024-01-01", "2024-12-31", max_retries=2)


class BackoffConfigEnvTests(unittest.TestCase):
    def test_from_env_reads_overrides(self) -> None:
        env = {
            "EMBEDDING_BACKOFF_BASE_SEC": "2.5",
            "EMBEDDING_BACKOFF_CAP_SEC": "120.0",
            "EMBEDDING_BACKOFF_JITTER": "equal",
            "EMBEDDING_BACKOFF_MAX_RETRIES": "5",
        }
        with patch.dict("os.environ", env, clear=True):
            cfg = BackoffConfig.from_env()
        self.assertEqual(cfg.base_sec, 2.5)
        self.assertEqual(cfg.cap_sec, 120.0)
        self.assertEqual(cfg.jitter, "equal")
        self.assertEqual(cfg.max_retries, 5)

    def test_from_env_falls_back_to_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            cfg = BackoffConfig.from_env()
        self.assertEqual(cfg.base_sec, 1.0)
        self.assertEqual(cfg.cap_sec, 60.0)
        self.assertEqual(cfg.jitter, "full")
        self.assertEqual(cfg.max_retries, 3)


if __name__ == "__main__":
    unittest.main()
