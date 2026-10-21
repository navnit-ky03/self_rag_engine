"""Offline API contract tests using an injected v3 app; no external calls."""
from __future__ import annotations

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class V3APIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prevent import-time LLM setup, actual retrieval, or network access.
        cls.fake_graph = MagicMock()
        cls.fake_graph.invoke.return_value = {"answer": "Grounded test answer."}
        with patch("config_v3.V3_ENABLED", True), patch.dict(
            sys.modules, {"graph_v3": SimpleNamespace(app=cls.fake_graph)}
        ):
            # Running in a fresh unittest process guarantees main is not
            # already bound to a different runtime.
            cls.main = importlib.import_module("main")
        from fastapi.testclient import TestClient
        cls.client = TestClient(cls.main.app)

    def setUp(self):
        self.fake_graph.reset_mock(return_value=True, side_effect=True)
        self.fake_graph.invoke.return_value = {"answer": "Grounded test answer."}

    def test_health_and_api_contract(self):
        health = self.client.get("/")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "online")
        response = self.client.post("/chat", json={
            "question": "Tell me about NeuroScan",
            "thread_id": "v3-contract-1",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": "Grounded test answer."})
        args, kwargs = self.fake_graph.invoke.call_args
        self.assertEqual(args[0], {"question": "Tell me about NeuroScan"})
        self.assertEqual(kwargs["config"]["configurable"]["thread_id"], "v3-contract-1")

    def test_chat_requires_thread_id(self):
        response = self.client.post("/chat", json={"question": "Hello"})
        self.assertEqual(response.status_code, 422)

    def test_upstream_failure_returns_503_without_leaking_exception(self):
        self.fake_graph.invoke.side_effect = RuntimeError("SECRET-UPSTREAM-DETAIL")
        response = self.client.post("/chat", json={
            "question": "Tell me about NeuroScan",
            "thread_id": "v3-contract-failure",
        })
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("SECRET-UPSTREAM-DETAIL", response.text)
        self.assertIn("temporarily unavailable", response.text)

    def test_v3_stream_returns_only_verified_final_answer(self):
        response = self.client.post("/chat/stream", json={
            "question": "Tell me about NeuroScan",
            "thread_id": "v3-contract-stream",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("Grounded test answer.", response.text)
        self.assertIn("data: [DONE]", response.text)
        self.assertNotIn("draft", response.text.lower())


if __name__ == "__main__":
    unittest.main()
