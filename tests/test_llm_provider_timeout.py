"""Tests for the OpenAI client timeout/max_retries config (Fix #1).

The hang observed in production ("tool node stuck for 2 minutes after a
multi-year Azure query") was traced to a sync OpenAI client with no
timeout. This module ensures the provider is built with a sane default
(30s) and that the env override is honored.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class LLMProviderTimeoutTests(unittest.TestCase):
    """Fix #1 — OpenAI sync + async clients must carry a request timeout."""

    def _build_provider(self):
        from rag.llm_provider import LLMProvider

        cfg = SimpleNamespace(
            provider="github_models",
            api_key="sk-test",
            base_url="https://models.inference.ai.azure.com",
            embedding_api_key="sk-test",
            embedding_base_url="https://models.inference.ai.azure.com",
            chat_model="gpt-4.1-mini",
            embedding_model="text-embedding-3-small",
        )
        return LLMProvider(cfg)

    def test_default_request_timeout_is_30_seconds(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_REQUEST_TIMEOUT", None)
            os.environ.pop("OPENAI_MAX_RETRIES", None)
            prov = self._build_provider()
            self.assertEqual(prov.request_timeout, 30.0)
            self.assertEqual(prov.max_retries, 2)

    def test_env_var_overrides_request_timeout(self):
        with patch.dict(os.environ, {"OPENAI_REQUEST_TIMEOUT": "12.5"}):
            prov = self._build_provider()
            self.assertEqual(prov.request_timeout, 12.5)

    def test_env_var_overrides_max_retries(self):
        with patch.dict(os.environ, {"OPENAI_MAX_RETRIES": "4"}):
            prov = self._build_provider()
            self.assertEqual(prov.max_retries, 4)

    def test_sync_client_carries_timeout(self):
        from openai import OpenAI

        with patch.dict(os.environ, {"OPENAI_REQUEST_TIMEOUT": "15.0"}):
            prov = self._build_provider()
        self.assertIsInstance(prov.client, OpenAI)
        # httpx.Timeout exposes the seconds we passed in
        timeout = prov.client.timeout
        # httpx.Timeout may wrap connect/read/write/pool — we only need
        # the read or total seconds to match our intent.
        self.assertEqual(float(timeout), 15.0)

    def test_async_client_carries_timeout(self):
        from openai import AsyncOpenAI

        with patch.dict(os.environ, {"OPENAI_REQUEST_TIMEOUT": "25.0"}):
            prov = self._build_provider()
        self.assertIsInstance(prov.async_client, AsyncOpenAI)
        self.assertEqual(float(prov.async_client.timeout), 25.0)

    def test_embedding_client_carries_timeout(self):
        from openai import OpenAI

        with patch.dict(os.environ, {"OPENAI_REQUEST_TIMEOUT": "20.0"}):
            prov = self._build_provider()
        self.assertIsInstance(prov.embedding_client, OpenAI)
        self.assertEqual(float(prov.embedding_client.timeout), 20.0)

    def test_invalid_timeout_falls_back_to_default(self):
        with patch.dict(os.environ, {"OPENAI_REQUEST_TIMEOUT": "not-a-number"}):
            prov = self._build_provider()
            self.assertEqual(prov.request_timeout, 30.0)

    def test_invalid_max_retries_falls_back_to_default(self):
        with patch.dict(os.environ, {"OPENAI_MAX_RETRIES": "not-a-number"}):
            prov = self._build_provider()
            self.assertEqual(prov.max_retries, 2)


if __name__ == "__main__":
    unittest.main()
